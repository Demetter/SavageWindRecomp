#include "perf_survey.h"
#include "kernel.h"
#include <stdlib.h>
#include <string.h>

int perf_survey_mode;
_Atomic uint64_t perf_survey_frame;
_Atomic int perf_survey_block = -1, perf_survey_enabled;
uint64_t perf_survey_epoch, perf_survey_start_ns = 45000000000ull;
static uint64_t block_ns = 10000000000ull;
static const char *names[] = {"off", "main", "jobs", "producer", "descriptor", "recorder", "gpu", "stub", "cpu-fence", "overlap-sensitivity", "overlap-jobs-stub", "overlap-queue-ahead", "overlap-downstream", "job-window", "fence-flags", "safe-overlap"};

void perf_survey_init(void) {
    const char *v = getenv("SAVAGEWIND_SURVEY");
    if (!v || !*v) return;
    for (int i = 1; i <= SURVEY_SAFE_OVERLAP; i++) if (!strcmp(v, names[i])) perf_survey_mode = i;
    if (!perf_survey_mode) FATAL("Unknown SAVAGEWIND_SURVEY mode: %s\n", v);
    const char *s = getenv("SAVAGEWIND_SURVEY_START");
    if (s && atof(s) >= 10) perf_survey_start_ns = (uint64_t)(atof(s) * 1e9);
    perf_survey_epoch = plat_now_ns();
    if (perf_survey_mode == SURVEY_SAFE_OVERLAP)
        LOG("[survey] safe overlap OFF/ON ABBA; descriptor completion plus guest-memory retirement required; normal GPU wait on fallback\n");
    if (perf_survey_mode == SURVEY_FENCE_FLAGS)
        LOG("[survey] normal GPU fences; stale-flag guard OFF/ON ABBA; CPU-completion diagnostic disabled\n");
    if (perf_survey_mode == SURVEY_DOWNSTREAM)
        LOG("[survey] CPU-completion fence held ON; blocks0-3 recorder delay ABBA; blocks4-7 descriptor delay ABBA\n");
    if (perf_survey_mode == SURVEY_JOB_WINDOW)
        LOG("[survey] job capacity OFF=512 ON=4096 ABBA; normal fences blocks0-3; CPU-completion fences blocks4-7; switch only at drained list boundary\n");
    if (perf_survey_mode == SURVEY_WAIT_JOBS)
        LOG("[survey] CPU-completion fence held ON; blocks0-7 job delay ABBA; blocks8-11 renderer-stub ABBA\n");
    if (perf_survey_mode == SURVEY_QUEUE_AHEAD)
        LOG("[survey] CPU-completion fence held ON; present queue limit OFF=1 ON=2, ABBA\n");
    if (perf_survey_mode == SURVEY_OVERLAP_SENSITIVITY)
        LOG("[survey] CPU-completion fence held ON for all measurement blocks; blocks0-3 producer delay ABBA, blocks4-7 main delay ABBA, blocks8-11 jobs delay ABBA\n");
    LOG("[survey] DIAGNOSTIC ONLY target %s start_ns %llu block_ns %llu delay_ns 500000 sequence ABBA; never promote\n",
        v, (unsigned long long)perf_survey_start_ns, (unsigned long long)block_ns);
}

void perf_survey_present(void) {
    if (!perf_survey_mode) return;
    uint64_t now = plat_now_ns(), elapsed = now - perf_survey_epoch;
    int b = elapsed < perf_survey_start_ns ? -1 : (int)((elapsed - perf_survey_start_ns) / block_ns);
    int enabled = b >= 0 && (b % 4 == 1 || b % 4 == 2);
    atomic_store_explicit(&perf_survey_block, b, memory_order_relaxed);
    atomic_store_explicit(&perf_survey_enabled, enabled, memory_order_relaxed);
    uint64_t f = atomic_fetch_add_explicit(&perf_survey_frame, 1, memory_order_relaxed) + 1;
    TRACE_LOG("[survey-frame] %llu %llu %d %d\n", (unsigned long long)now, (unsigned long long)f, b, enabled);
}

void perf_survey_delay(int role, unsigned identity) {
    if (perf_survey_mode == SURVEY_DOWNSTREAM) {
        int b = atomic_load_explicit(&perf_survey_block, memory_order_relaxed);
        if (role != (b < 4 ? SURVEY_RECORDER : SURVEY_DESCRIPTOR)) return;
    } else if (perf_survey_mode == SURVEY_WAIT_JOBS) {
        if (role != SURVEY_JOBS || atomic_load_explicit(&perf_survey_block, memory_order_relaxed) >= 8) return;
    } else if (perf_survey_mode == SURVEY_OVERLAP_SENSITIVITY) {
        int b = atomic_load_explicit(&perf_survey_block, memory_order_relaxed);
        if (role != (b < 4 ? SURVEY_PRODUCER : b < 8 ? SURVEY_MAIN : SURVEY_JOBS)) return;
    } else if (perf_survey_mode != role) return;
    static __thread uint64_t last_frame, count, sum, maximum;
    static __thread int old_block = -2;
    uint64_t frame = atomic_load_explicit(&perf_survey_frame, memory_order_relaxed);
    if (!frame || frame == last_frame) return;
    last_frame = frame;
    int b = atomic_load_explicit(&perf_survey_block, memory_order_relaxed);
    if (b != old_block) {
        if (old_block >= 0) TRACE_LOG("[survey-dose] role %s id %x block %d count %llu ns %llu max_ns %llu\n",
            names[role], identity, old_block, (unsigned long long)count, (unsigned long long)sum, (unsigned long long)maximum);
        count = sum = maximum = 0; old_block = b;
    }
    if (!perf_survey_on(role)) return;
    uint64_t start = plat_now_ns(), end;
    do { __asm__ volatile("pause"); end = plat_now_ns(); } while (end - start < 500000);
    uint64_t dt = end - start; count++; sum += dt; if (dt > maximum) maximum = dt;
}

void perf_survey_guest(KThread *t) {
    if (!t || t->core_held < 0 || plat_now_ns() - perf_survey_epoch < 20000000000ull) return;
    static __thread uint32_t identified;
    static __thread int role;
    if (identified != t->handle) {
        char name[96] = "";
        extern bool kernel_survey_thread_name(KThread *, char *, size_t);
        if (!kernel_survey_thread_name(t, name, sizeof name)) return;
        identified = t->handle;
        role = t->handle == 0x101 ? SURVEY_MAIN :
            !strcmp(name, "Worker") ? SURVEY_JOBS : SURVEY_OFF;
        TRACE_LOG("[survey-thread] handle %x core %d priority %d role %s name %s entry %llx\n",
            t->handle, t->core, t->base_prio, names[role], name, (unsigned long long)t->entry);
    }
    if (role) perf_survey_delay(role, t->handle);
}
