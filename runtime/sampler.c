// (profiling) Sampling profiler for Windows. SAVAGEWIND_SAMPLE=<start s>:<seconds>[:<interval us>] suspends every
// guest thread each interval (default 1000 us) from <start> seconds after boot, for <seconds>, and records its host
// call stack (x64 unwind data: the game library and the runtime both have it). The samples go to
// build/samples.txt; tools/profile_samples.py turns them into self / inclusive time per function, recompiled game
// functions as f_<guest address>. A thread waiting in the kernel shows its wait; filter with the tool.
// Each round of samples starts with "R <ns> <presented frame>", so a stretch of frames can be looked at alone.
#include <stdlib.h>
#include <string.h>

#include "kernel.h"
#include "rt.h"

#ifdef _WIN32
#include <windows.h>
#include <tlhelp32.h>
#include "platform/sample_unwind.h"

#define MAX_THREADS 128
#define MAX_DEPTH 48

// These workers live for the process lifetime. Duplicate the current native
// handle, independently of whether the creator has already detached pthread_t.
static HANDLE sample_host_handles[4];
void sampler_register_host_thread(unsigned index) {
    if (index >= 4 || !getenv("SAVAGEWIND_SAMPLE")) return;
    HANDLE handle = NULL;
    if (DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(),
                        &handle, 0, FALSE, DUPLICATE_SAME_ACCESS)) sample_host_handles[index] = handle;
}

static void sample_modules(FILE *f) {
    HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPMODULE, GetCurrentProcessId());
    if (snapshot == INVALID_HANDLE_VALUE) return;
    MODULEENTRY32 module = {.dwSize = sizeof module};
    if (Module32First(snapshot, &module)) do {
        fprintf(f, "D %llx %lx %s\n", (unsigned long long)(uintptr_t)module.modBaseAddr,
                (unsigned long)module.modBaseSize, module.szExePath);
    } while (Module32Next(snapshot, &module));
    CloseHandle(snapshot);
}

// Use committed readable stack bounds, not DeallocationStack (the reservation
// includes guard and uncommitted pages). The target is suspended while read.
typedef struct { LONG ExitStatus; PVOID TebBaseAddress; ULONG_PTR ClientId[2]; ULONG_PTR AffinityMask; LONG Priority, BasePriority; } TBI;
static bool thread_stack(HANDLE h, uint64_t sp, uint64_t *lo, uint64_t *hi) {
    typedef LONG(NTAPI * Q)(HANDLE, int, PVOID, ULONG, PULONG);
    static Q q;
    if (!q) q = (Q)(void *)GetProcAddress(GetModuleHandleA("ntdll.dll"), "NtQueryInformationThread");
    TBI tbi;
    if (!q || q(h, 0, &tbi, sizeof tbi, NULL) != 0 || !tbi.TebBaseAddress) return false;
    const uint8_t *teb = tbi.TebBaseAddress;
    *hi = *(const uint64_t *)(teb + 8);
    *lo = *(const uint64_t *)(teb + 16); // NT_TIB.StackLimit
    MEMORY_BASIC_INFORMATION region;
    if (VirtualQuery((void *)(uintptr_t)sp, &region, sizeof region) != sizeof region ||
        region.State != MEM_COMMIT || (region.Protect & (PAGE_GUARD | PAGE_NOACCESS))) return false;
    uint64_t begin = (uint64_t)(uintptr_t)region.BaseAddress, end = begin + region.RegionSize;
    if (*lo < begin) *lo = begin;
    if (*hi > end) *hi = end;
    return *hi > *lo;
}

static void *sampler_main(void *arg) {
    double start = 0, secs = 0, us = 1000;
    sscanf((const char *)arg, "%lf:%lf:%lf", &start, &secs, &us);
    if (secs <= 0) return NULL;
    if (!sample_unwind_init()) LOG("[sample] bounded unwinder unavailable; recording instruction pointers only\n");
    uint64_t due = plat_now_ns() + (uint64_t)(start * 1e9);
    uint32_t selected[16];
    unsigned selected_n = 0;
    const char *filter = getenv("SAVAGEWIND_SAMPLE_HANDLES");
    for (const char *p = filter; p && *p;) {
        char *end;
        unsigned long value = strtoul(p, &end, 0);
        if (end == p || (*end && *end != ',') || selected_n == 16 || !value || value > UINT32_MAX) {
            LOG("[sample] invalid handle filter; sampling disabled\n"); return NULL;
        }
        selected[selected_n++] = (uint32_t)value;
        p = *end ? end + 1 : end;
    }
    const char *mb_text = getenv("SAVAGEWIND_SAMPLE_BUFFER_MB");
    unsigned mb = mb_text ? (unsigned)atoi(mb_text) : 96;
    if (mb < 16 || mb > 512) mb = 96;
    size_t cap = (size_t)mb * 1024 * 1024 / sizeof(uint64_t), used = 0;
    // Allocate and prefault during startup, before the sampled window. Nothing
    // may allocate while a thread is suspended, including first-touch faults.
    uint64_t *buf = malloc(cap * sizeof *buf);
    if (!buf) return NULL;
    for (size_t i = 0; i < cap; i += 4096 / sizeof *buf) ((volatile uint64_t *)buf)[i] = 0;
    plat_sleep_until_ns(due);
    pthread_t host[MAX_THREADS];
    uint32_t handle[MAX_THREADS];
    uint64_t entry[MAX_THREADS];
    bool host_only = getenv("SAVAGEWIND_SAMPLE_HOST_ONLY") && atoi(getenv("SAVAGEWIND_SAMPLE_HOST_ONLY"));
    uint64_t begin = plat_now_ns(), end = begin + (uint64_t)(secs * 1e9), nsamples = 0;
    uint64_t first_round = 0, last_round = 0, rounds = 0;
    bool full = false;
    LOG("[sample] sampling every %.0f us for %.0f s (%s), buffer %u MB, handle filter %s\n", us, secs,
        host_only ? "renderer host threads" : "registered threads", mb, filter ? filter : "all");
    for (uint64_t next = plat_now_ns(); plat_now_ns() < end; next += (uint64_t)(us * 1000)) {
        if (used + MAX_DEPTH + 8 >= cap) { full = true; break; }
        int n = kernel_running_threads(host, handle, entry, MAX_THREADS);
        if (used + 4 < cap) {  // a round marker: "R <ns> <presented frame>" in the file
            extern uint64_t renderer_present_count(void);
            buf[used++] = ~0ull;
            buf[used++] = 2;
            last_round = plat_now_ns();
            if (!first_round) first_round = last_round;
            rounds++;
            buf[used++] = last_round;
            buf[used++] = renderer_present_count();
        }
        for (int i = 0; i < n && used + MAX_DEPTH + 4 < cap; i++) {
            if (host_only && handle[i] < 0xf000) continue;
            if (selected_n) {
                bool found = false;
                for (unsigned j = 0; j < selected_n; j++) found |= selected[j] == handle[i];
                if (!found) continue;
            }
            HANDLE h = handle[i] >= 0xf000 && handle[i] < 0xf004 ? sample_host_handles[handle[i] - 0xf000] :
                       (HANDLE)pthread_gethandle(host[i]);
            if (!h || SuspendThread(h) == (DWORD)-1) continue;
            CONTEXT ctx;
            memset(&ctx, 0, sizeof ctx);
            ctx.ContextFlags = CONTEXT_FULL;
            if (GetThreadContext(h, &ctx)) {
                uint64_t slo = 0, shi = 0;
                bool valid_stack = thread_stack(h, ctx.Rsp, &slo, &shi);
                size_t at = used;
                buf[used++] = handle[i];
                buf[used++] = 0;  // depth, filled below
                int d = 0;
                while (d < MAX_DEPTH && ctx.Rip) {
                    buf[used++] = ctx.Rip;
                    d++; // The saved PC must count even if this frame cannot unwind.
                    if (!valid_stack || !sample_unwind_step(&ctx, slo, shi)) break;
                }
                buf[at + 1] = (uint64_t)d;
                nsamples++;
            }
            ResumeThread(h);
        }
        uint64_t now = plat_now_ns();
        if (next > now) plat_sleep_until_ns(next);
        else next = now;
    }
    FILE *f = fopen("build/samples.txt", "w");
    if (f) {
        fprintf(f, "P %llu %llu %llu %llu %llu %u\n", (unsigned long long)begin, (unsigned long long)end,
                (unsigned long long)first_round, (unsigned long long)last_round, (unsigned long long)rounds, full);
        fprintf(f, "M exe %llx\nM swgame %llx\n", (unsigned long long)(uintptr_t)GetModuleHandleA(NULL),
                (unsigned long long)(uintptr_t)GetModuleHandleA("swgame.dll"));
        sample_modules(f);
        int n = kernel_running_threads(host, handle, entry, MAX_THREADS);
        for (int i = 0; i < n; i++) {
            fprintf(f, "T %u %llx\n", handle[i], (unsigned long long)entry[i]);
            HANDLE h = handle[i] >= 0xf000 && handle[i] < 0xf004 ? sample_host_handles[handle[i] - 0xf000] :
                       (HANDLE)pthread_gethandle(host[i]);
            if (h) fprintf(f, "H %u %lu\n", handle[i], GetThreadId(h));
        }
        for (size_t p = 0; p < used;) {
            uint64_t th = buf[p], d = buf[p + 1];
            if (th == ~0ull) {
                fprintf(f, "R %llu %llu\n", (unsigned long long)buf[p + 2], (unsigned long long)buf[p + 3]);
                p += 2 + d;
                continue;
            }
            fprintf(f, "S %llu", (unsigned long long)th);
            for (uint64_t k = 0; k < d; k++) fprintf(f, " %llx", (unsigned long long)buf[p + 2 + k]);
            fputc('\n', f);
            p += 2 + d;
        }
        fclose(f);
    }
    LOG("[sample] %llu samples written to build/samples.txt\n", (unsigned long long)nsamples);
    LOG("[sample] coverage %.3f of %.3f requested seconds; %llu rounds%s\n",
        first_round ? (last_round-first_round)/1e9 : 0, (end-begin)/1e9, (unsigned long long)rounds,
        full ? "; BUFFER EXHAUSTED: later stutters have no stacks" : "");
    free(buf);
    return NULL;
}

__attribute__((constructor)) static void sampler_start(void) {
    const char *e = getenv("SAVAGEWIND_SAMPLE");
    if (!e) return;
    pthread_t th;
    pthread_create(&th, NULL, sampler_main, (void *)e);
    pthread_detach(th);
}
#endif
