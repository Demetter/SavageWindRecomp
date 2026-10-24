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

#define MAX_THREADS 128
#define MAX_DEPTH 48

// The thread's stack (its TEB: NT_TIB StackBase, and DeallocationStack at +0x1478 on x64), so the unwinder stops at a
// frame whose stack pointer lies outside it: RtlVirtualUnwind read through a bad one and crashed the game.
typedef struct { LONG ExitStatus; PVOID TebBaseAddress; ULONG_PTR ClientId[2]; ULONG_PTR AffinityMask; LONG Priority, BasePriority; } TBI;
static bool thread_stack(HANDLE h, uint64_t *lo, uint64_t *hi) {
    typedef LONG(NTAPI * Q)(HANDLE, int, PVOID, ULONG, PULONG);
    static Q q;
    if (!q) q = (Q)(void *)GetProcAddress(GetModuleHandleA("ntdll.dll"), "NtQueryInformationThread");
    TBI tbi;
    if (!q || q(h, 0, &tbi, sizeof tbi, NULL) != 0 || !tbi.TebBaseAddress) return false;
    const uint8_t *teb = tbi.TebBaseAddress;
    *hi = *(const uint64_t *)(teb + 8);
    *lo = *(const uint64_t *)(teb + 0x1478);
    return *hi > *lo;
}

static void *sampler_main(void *arg) {
    double start = 0, secs = 0, us = 1000;
    sscanf((const char *)arg, "%lf:%lf:%lf", &start, &secs, &us);
    if (secs <= 0) return NULL;
    plat_sleep_until_ns(plat_now_ns() + (uint64_t)(start * 1e9));
    size_t cap = 12u << 20, used = 0;  // words (96 MB), allocated up front: nothing may allocate while a thread is suspended
    uint64_t *buf = malloc(cap * sizeof *buf);
    if (!buf) return NULL;
    pthread_t host[MAX_THREADS];
    uint32_t handle[MAX_THREADS];
    uint64_t entry[MAX_THREADS];
    uint64_t end = plat_now_ns() + (uint64_t)(secs * 1e9), nsamples = 0;
    LOG("[sample] sampling every %.0f us for %.0f s\n", us, secs);
    for (uint64_t next = plat_now_ns(); plat_now_ns() < end; next += (uint64_t)(us * 1000)) {
        int n = kernel_running_threads(host, handle, entry, MAX_THREADS);
        if (used + 4 < cap) {  // a round marker: "R <ns> <presented frame>" in the file
            extern uint64_t renderer_present_count(void);
            buf[used++] = ~0ull;
            buf[used++] = 2;
            buf[used++] = plat_now_ns();
            buf[used++] = renderer_present_count();
        }
        for (int i = 0; i < n && used + MAX_DEPTH + 4 < cap; i++) {
            HANDLE h = (HANDLE)pthread_gethandle(host[i]);
            if (!h || SuspendThread(h) == (DWORD)-1) continue;
            CONTEXT ctx;
            memset(&ctx, 0, sizeof ctx);
            ctx.ContextFlags = CONTEXT_FULL;
            uint64_t slo = 0, shi = ~0ull;
            if (!thread_stack(h, &slo, &shi)) slo = 0, shi = ~0ull;
            if (GetThreadContext(h, &ctx)) {
                size_t at = used;
                buf[used++] = handle[i];
                buf[used++] = 0;  // depth, filled below
                int d = 0;
                for (; d < MAX_DEPTH && ctx.Rip; d++) {
                    buf[used++] = ctx.Rip;
                    if (ctx.Rsp < slo || ctx.Rsp + 8 > shi) { d++; break; }
                    DWORD64 base = 0;
                    PRUNTIME_FUNCTION f = RtlLookupFunctionEntry(ctx.Rip, &base, NULL);
                    if (!f) {  // a leaf without unwind data: the return address is on top of the stack
                        if (!ctx.Rsp || IsBadReadPtr((void *)ctx.Rsp, 8)) break;
                        ctx.Rip = *(DWORD64 *)ctx.Rsp;
                        ctx.Rsp += 8;
                        continue;
                    }
                    {   // a frame register (UNWIND_INFO FrameRegister: CONTEXT's Rax.. order) outside the stack: stop there
                        const uint8_t *ui = (const uint8_t *)(uintptr_t)(base + f->UnwindData);
                        int freg = ui[3] & 0xF;
                        DWORD64 fv = freg ? (&ctx.Rax)[freg] : slo;
                        if (fv < slo || fv >= shi) { d++; break; }
                    }
                    void *hd = NULL;
                    DWORD64 fr = 0;
                    RtlVirtualUnwind(UNW_FLAG_NHANDLER, base, ctx.Rip, f, &ctx, &hd, &fr, NULL);
                    if (ctx.Rsp < slo || ctx.Rsp >= shi) { d++; break; }
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
        fprintf(f, "M exe %llx\nM swgame %llx\n", (unsigned long long)(uintptr_t)GetModuleHandleA(NULL),
                (unsigned long long)(uintptr_t)GetModuleHandleA("swgame.dll"));
        int n = kernel_running_threads(host, handle, entry, MAX_THREADS);
        for (int i = 0; i < n; i++) fprintf(f, "T %u %llx\n", handle[i], (unsigned long long)entry[i]);
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
