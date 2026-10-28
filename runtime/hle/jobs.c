// BOTW's job scheduler, run natively (BOTW-JOBS-NATIVE.md step 2). The game pushes each job group into a ring per sead
// job worker its core mask names, wakes the workers through their thread message queues and waits for each worker's
// done event: the Switch's three-core scheduler over kernel objects. Here a pool's round list (the groups pushed since
// its last run-and-wait, in push order; the game keeps it at pool + 0x90) is the one queue. Workers walk it and run
// their share of each group whose mask has their index, through the group's own run-chunk, which claims jobs from the
// group's shared counter; idle workers and a barrier group's waiters block on the OS's address wait. The main thread
// runs index 0's share in run-and-wait, then waits for every group of the round. Job workers run as ordinary OS
// threads, off the emulated core tokens. The game's group code (setup, reset, run-chunk, done check, event) is called
// as it is, and the workers' state words keep the values the game writes.
//
// SAVAGEWIND_JOBS_NATIVE=0: the game's own scheduler. SAVAGEWIND_JOBS_SPIN=<n>: pauses before a wait blocks (1000).
// (check) SAVAGEWIND_JOBS_CHECK=1: at each round's end every group's completions must equal its job count and (for the
// base group class) its claim counter must have reached it - no job run twice, none left.
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "../cpu.h"
#include "../hle.h"
#include "../kernel.h"
#include "../platform.h"
#include "../rt.h"
#include "../perf_survey.h"
#include "../nvn/nvn.h"  // (NvnFnStat: the layer Draw jobs' NVN calls by function)

#if defined(__x86_64__)
#include <immintrin.h>
#define PAUSE() _mm_pause()
#else
#define PAUSE() __asm__ volatile("yield")
#endif

// guest layouts (game 1.9.0)
enum {
    POOL_COUNT = 0x78, POOL_WORKERS = 0x80, POOL_LIST_CAP = 0x88, POOL_LIST = 0x90, POOL_LIST_N = 0x98,
    POOL_SERIAL = 0x9c, POOL_KICKS = 0xa0, POOL_KICK_TICK = 0xa8,
    W_ID = 0xd0 /* sead::Thread's id */, W_INDEX = 0xfc, W_STATE = 0x100, W_GROUP = 0x130, W_OWNER = 0x138, W_RUNS = 0x140, W_RUN_TICK = 0x148,
    G_TYPE = 0x18, G_MASK = 0x20, G_EVENT = 0x28, G_GRAB = 0x58, G_FLAGS = 0x64, G_DONE = 0x70, G_OWNER = 0x78,
    G_JOBS = 0x90, G_NEXT = 0x94,
    V_RESET = 0x08, V_RUN_CHUNK = 0x10, V_TOTAL = 0x18, V_IS_DONE = 0x20, V_SEND = 0x18,
};
// the game functions the hooks call (tools/patches.py passes them: group setup, tick, group event clear / signal)
enum { F_SETUP, F_TICK, F_CLEAR, F_SIGNAL };

typedef struct {
    uint64_t pool;  // (0: free; set once, with release)
    uint32_t seq;   // workers' wait word: a round opens, a group joins an open round, the round closes
    uint32_t evt;   // run-and-wait's wait word: a group finishes, a group joins an open round
    uint32_t open;  // kicked and not yet closed by run-and-wait
    uint32_t gen;   // rounds closed
    uint32_t busy;  // workers walking the round list
    uint32_t fan;   // the seq a woken worker last passed the wake-up on for (wake_workers)
    uint8_t started[32];  // worker i's thread is in the native loop
} Pool;
static Pool g_pools[8];
static pthread_mutex_t g_pools_m = PTHREAD_MUTEX_INITIALIZER;
static int g_on = -1, g_check, g_spin;
static int g_spin_alt = -1;  // (PERFX_JSPIN: the spin in the on blocks, SAVAGEWIND_JOBS_SPIN_ALT, default 50)
static uint64_t g_rounds, g_groups, g_bad;

RT_ON_RESET(jobs_reboot_reset) { memset(g_pools, 0, sizeof g_pools); }  // (reboot: pools and threads were the guest's)

// (research) SAVAGEWIND_JOBS_STATS=1: where the rounds' time goes, a summary every 2 s of rounds, per second - by
// group size (jobs: 1, 2-4, 5-8, 9-16, 17-48, 49+) the groups, their chunk size, the threads' time on them, the span
// from a group's first claim to its completion and how many threads ran some of it; per round the main thread's time
// running its share and waiting, the span from kick to run-and-wait's end, the workers' waits on barrier groups.
enum { SB = 6 };
static int g_stats;
static struct { uint64_t groups, jobs, grab, thread, span, parts; } g_sb[SB];
static uint64_t g_st_rounds, g_st_main_run, g_st_main_wait, g_st_span, g_st_barrier, g_st_t0, g_st_kick;
static uint64_t g_st_first[64];  // (a round-list slot's first claim; 0: none yet)
// (research) SAVAGEWIND_JOBS_TRACE=<hex>: groups whose first job's class ends in those hex digits, traced: each
// thread's entry and exit (us from the round's kick, jobs it ran) and the group's completion, for 40 groups.
static uint64_t g_tr_key, g_tr_mask;
static int g_tr_left = 40;
static bool trace_group(uint64_t g, uint32_t total) {
    if (!g_tr_mask || __atomic_load_n(&g_tr_left, __ATOMIC_RELAXED) <= 0) return false;
    if (total != LD32(g + 0x90) || !LD32(g + 0x80) || !LD64(g + 0x88)) return false;
    uint64_t job0 = LD64(LD64(g + 0x88));
    return job0 && (LD64(job0) & g_tr_mask) == g_tr_key;
}
// (research) SAVAGEWIND_JOBS_ROUNDS=<first round>:<rounds>: every group visit of those rounds (any thread), kept in
// memory and written once they end ("[jobs-round]" lines: times in us from the round's kick), with each round's
// frame, its run-and-wait's start and its end: the rounds' timelines, for their critical path.
typedef struct { uint64_t g, kind; uint32_t round, slot, worker, type, total, jobs, done; float in, jd, out; } RoundEv;
typedef struct { uint64_t frame, pool; uint32_t round, ngroups; float wait_in, end; } RoundRec;
static RoundEv *g_rev;
static RoundRec *g_rrec;
static uint32_t g_rev_n, g_rrec_n;
static uint64_t g_round_no, g_rt_first, g_rt_count;
static bool round_traced(void) {
    uint64_t r = __atomic_load_n(&g_round_no, __ATOMIC_RELAXED);
    return g_rt_count && r >= g_rt_first && r < g_rt_first + g_rt_count;
}
static uint64_t job_kind(uint64_t g, uint32_t total) {  // (the first job's class, for a base group)
    if (total != LD32(g + 0x90) || !LD32(g + 0x80) || !LD64(g + 0x88)) return 0;
    uint64_t job0 = LD64(LD64(g + 0x88));
    return job0 ? LD64(job0) : 0;
}
// (research) SAVAGEWIND_JOBS_JOBTIME=1: every job's time (a run-chunk call that ran one job: grab 1), by the job's
// class and its two dispatch words (+0x18 and +0x20: a switch job's case, a delegate job's case and object), with the
// longest; every 2 s the 16 with the most time a second ("[jobs-time]").
static int g_jt;
typedef struct { uint64_t vt, w18, w20, n, ns, max, obj, nvn, calls, sn, swait, shold; } JobTime;
int g_nvn_time;                          // (runtime/nvn/nvn.h: NVN calls timed on their thread)
__thread uint64_t t_nvn_ticks, t_nvn_calls;
__thread int t_nvn_tag;                  // (a layer's Draw job running: its NVN calls counted by function)
static NvnFnStat *g_nvn_fns;
void nvn_fn_note(NvnFnStat *s, uint64_t ticks) {
    if (!__atomic_exchange_n(&s->listed, 1, __ATOMIC_ACQ_REL)) {
        NvnFnStat *h = __atomic_load_n(&g_nvn_fns, __ATOMIC_ACQUIRE);
        do s->next = h; while (!__atomic_compare_exchange_n(&g_nvn_fns, &h, s, false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE));
    }
    __atomic_add_fetch(&s->n, 1, __ATOMIC_RELAXED), __atomic_add_fetch(&s->ticks, ticks, __ATOMIC_RELAXED);
}
__thread uint64_t t_shadow_n, t_shadow_wait, t_shadow_hold;  // (runtime/nvn/cmdbuf.c: its records' lock)
static double g_tick_ns;                 // (ns per plat_ticks() tick)
#define JT_N 1024
static JobTime g_jtab[JT_N];
static pthread_mutex_t g_jt_m = PTHREAD_MUTEX_INITIALIZER;
static uint64_t g_jt_t0;
static int jobtime_is_draw(uint64_t g, uint32_t idx) {  // (1.9.0) a layer's Draw job: class 0x7001d07978, +0x18 0
    uint32_t cap = LD32(g + 0x80);
    uint64_t arr = LD64(g + 0x88);
    if (!cap || idx >= cap || idx >= LD32(g + G_JOBS) || arr < GUEST_LO || arr + 8ull * cap > GUEST_HI) return 0;
    uint64_t job = LD64(arr + 8ull * idx);
    if (job < GUEST_LO || job + 0x28 > GUEST_HI) return 0;
    return LD64(job) == 0x7001d07978ull && LD32(job + 0x18) == 0;
}
static void jobtime_note(uint64_t g, uint32_t idx, uint64_t ns, uint64_t ticks, uint64_t calls, uint64_t sn,
                         uint64_t swait, uint64_t shold) {
    // (a base group: its job array +0x88 holds +0x80 entries, its job count +0x90; every pointer checked as guest memory)
    uint32_t cap = LD32(g + 0x80);
    uint64_t arr = LD64(g + 0x88);
    if (!cap || idx >= cap || idx >= LD32(g + G_JOBS) || arr < GUEST_LO || arr + 8ull * cap > GUEST_HI) return;
    uint64_t job = LD64(arr + 8ull * idx);
    if (job < GUEST_LO || job + 0x28 > GUEST_HI || LD64(job) < GUEST_LO || LD64(job) >= GUEST_HI) return;
    uint64_t vt = LD64(job), w18 = LD32(job + 0x18), w20 = LD32(job + 0x20);
    if (vt == 0x7001d11e58ull && w20 == 4) {  // (1.9.0: the views' recording, f_7000fb5a20) its items and buckets
        static uint64_t last;
        uint64_t now = plat_now_ns(), obj = LD64(job + 0x18);
        if (now - last > 2000000000ull && obj >= GUEST_LO && obj < GUEST_HI) {
            last = now;
            uint32_t cap = LD32(obj + 0x5e40), made = LD32(obj + 0x5e54), cnt = LD32(obj + 0x5e58);
            uint64_t items = LD64(obj + 0x5e48), set = LD64(obj + 0x5e38);
            char buf[512]; int k = 0;
            for (uint32_t i = 0; i < cnt && i < cap && i < 24 && items >= GUEST_LO; i++)
                k += snprintf(buf + k, sizeof buf - k, " %u+%u", LD32(items + 40ull * i + 0x18), LD32(items + 40ull * i + 0x1c));
            LOG("[view-rec] items %u made %u cap %u, parts min %u max %u; ranges%s\n", cnt, made, cap,
                LD32(obj + 0x5e68), LD32(obj + 0x5e6c), buf);
            if (set >= GUEST_LO && set < GUEST_HI) {
                uint32_t nb = LD32(set + 0x50), bc = LD32(set + 0x40); uint64_t ba = LD64(set + 0x48);
                k = 0;
                for (uint32_t b = 0; b < nb && b < bc && b < 24 && ba >= GUEST_LO; b++)
                    k += snprintf(buf + k, sizeof buf - k, " %u", LD32(ba + 0x70ull * b + 0x58));
                LOG("[view-rec] buckets %u (cap %u), command buffers each:%s\n", nb, bc, buf);
            }
        }
    }
    uint64_t h = (vt * 0x9E3779B97F4A7C15ull) ^ (w18 * 0xC2B2AE3D27D4EB4Full) ^ w20;
    pthread_mutex_lock(&g_jt_m);
    for (uint32_t s = (uint32_t)(h >> 50) & (JT_N - 1), k = 0; k < JT_N; k++, s = (s + 1) & (JT_N - 1)) {
        JobTime *e = &g_jtab[s];
        if (!e->n) *e = (JobTime){vt, w18, w20, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        if (e->vt != vt || e->w18 != w18 || e->w20 != w20) continue;
        e->n++, e->ns += ns, e->nvn += (uint64_t)(ticks * g_tick_ns), e->calls += calls, e->sn += sn, e->swait += swait, e->shold += shold;
        if (ns > e->max) e->max = ns, e->obj = job;
        break;
    }
    uint64_t now = plat_now_ns();
    if (!g_jt_t0) g_jt_t0 = now;
    if (now - g_jt_t0 >= 2000000000ull) {
        double s = (now - g_jt_t0) / 1e9;
        bool shown[JT_N] = {0};
        {   // the layer Draw jobs' NVN calls by function, the most called first (counts since the start)
            NvnFnStat *top[24] = {0};
            for (NvnFnStat *f = __atomic_load_n(&g_nvn_fns, __ATOMIC_ACQUIRE); f; f = f->next)
                for (int k = 0; k < 24; k++)
                    if (!top[k] || f->n > top[k]->n) { memmove(top + k + 1, top + k, (23 - k) * sizeof top[0]); top[k] = f; break; }
            char buf[1600]; int k = 0;
            for (int i = 0; i < 24 && top[i] && k < (int)sizeof buf - 80; i++)
                k += snprintf(buf + k, sizeof buf - k, " %s %llu (%.0f us)", top[i]->name, (unsigned long long)top[i]->n,
                              top[i]->ticks * g_tick_ns / 1e3);
            if (k) LOG("[jobs-time] layer Draw jobs' NVN calls since the start:%s\n", buf);
        }
        LOG("[jobs-time] the jobs with the most time (class, +0x18, +0x20: a second, mean, longest)\n");
        for (int r = 0; r < 16; r++) {
            int b = -1;
            for (int k = 0; k < JT_N; k++)
                if (g_jtab[k].n && !shown[k] && (b < 0 || g_jtab[k].ns > g_jtab[b].ns)) b = k;
            if (b < 0) break;
            shown[b] = true;
            JobTime *e = &g_jtab[b];
            LOG("[jobs-time]   %010llx %8llx %8llx: %6.0f /s, %7.1f ms/s, mean %6.1f us, longest %6.1f us (job %llx); "
                "NVN calls %.0f a job, %.1f us of it; record lock %.1f a job, waited %.1f us, held %.1f us\n",
                (unsigned long long)e->vt, (unsigned long long)e->w18, (unsigned long long)e->w20, e->n / s,
                e->ns / 1e6 / s, e->ns / 1e3 / e->n, e->max / 1e3, (unsigned long long)e->obj,
                (double)e->calls / e->n, e->nvn / 1e3 / e->n, (double)e->sn / e->n, e->swait * g_tick_ns / 1e3 / e->n,
                e->shold * g_tick_ns / 1e3 / e->n);
        }
        memset(g_jtab, 0, sizeof g_jtab);
        g_jt_t0 = now;
    }
    pthread_mutex_unlock(&g_jt_m);
}

static int sbucket(uint32_t n) { return n <= 1 ? 0 : n <= 4 ? 1 : n <= 8 ? 2 : n <= 16 ? 3 : n <= 48 ? 4 : 5; }
#define SADD(v, x) __atomic_add_fetch(&(v), (x), __ATOMIC_RELAXED)
// ... and by group kind (its class, the first job's class, its mask's worker count, its type): the 16 kinds with the
// most threads' time; and the delay from a kick to each worker's start on the round
enum { SK = 128 };
static struct { uint64_t key, groups, jobs, thread, span, parts; uint32_t mask, type; } g_sk[SK];
static uint64_t g_st_wake, g_st_wakes;
static int skind(uint64_t g, uint32_t total) {
    // (a base group's jobs: count +0x80, pointers +0x88, each job's class its first word)
    uint64_t jobs = total == LD32(g + G_JOBS) && LD32(g + 0x80) ? LD64(g + 0x88) : 0, job0 = jobs ? LD64(jobs) : 0;
    uint64_t first = job0 ? LD64(job0) : 0;
    uint64_t key = (LD64(g) & 0xffffffffffull) << 24 ^ (first & 0xffffffffffull) ^ (uint64_t)LD32(g + G_TYPE) << 60;
    for (uint32_t h = (uint32_t)(key * 0x9E3779B97F4A7C15ull >> 57), i = 0; i < SK; i++, h = (h + 1) & (SK - 1)) {
        uint64_t k = __atomic_load_n(&g_sk[h].key, __ATOMIC_RELAXED), z = 0;
        if (k == key) return (int)h;
        if (!k && __atomic_compare_exchange_n(&g_sk[h].key, &z, key, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
            g_sk[h].mask = LD32(g + G_MASK), g_sk[h].type = LD32(g + G_TYPE);
            return (int)h;
        }
        if (z == key) return (int)h;
    }
    return -1;
}

static uint32_t *P32(uint64_t a) { return (uint32_t *)(uintptr_t)a; }
static uint64_t vfn(uint64_t obj, uint32_t off) { return LD64(LD64(obj) + off); }
static uint64_t call1(Cpu *c, uint64_t fn, uint64_t a0) {
    c->x[0] = a0;
    guest_call(c, fn);
    return c->x[0];
}

static bool on(void) {
    if (g_on < 0) {
        const char *e = getenv("SAVAGEWIND_JOBS_NATIVE"), *s = getenv("SAVAGEWIND_JOBS_SPIN");
        g_check = getenv("SAVAGEWIND_JOBS_CHECK") != NULL;
        g_stats = getenv("SAVAGEWIND_JOBS_STATS") != NULL;
        g_jt = getenv("SAVAGEWIND_JOBS_JOBTIME") != NULL;
        if (g_jt) {  // (the tick rate, against the clock over 20 ms)
            uint64_t n0 = plat_now_ns(), k0 = plat_ticks();
            while (plat_now_ns() - n0 < 20000000ull) {}
            g_tick_ns = (double)(plat_now_ns() - n0) / (double)(plat_ticks() - k0);
            g_nvn_time = 1;
        }
        const char *rt = getenv("SAVAGEWIND_JOBS_ROUNDS");
        if (rt && sscanf(rt, "%llu:%llu", (unsigned long long *)&g_rt_first, (unsigned long long *)&g_rt_count) == 2 && g_rt_count) {
            g_rev = calloc(400000, sizeof *g_rev);
            g_rrec = calloc(g_rt_count, sizeof *g_rrec);
        }
        const char *tr = getenv("SAVAGEWIND_JOBS_TRACE");
        if (tr && *tr) {
            g_tr_key = strtoull(tr, NULL, 16);
            g_tr_mask = strlen(tr) >= 16 ? ~0ull : (1ull << (4 * strlen(tr))) - 1;
        }
        g_spin = s ? atoi(s) : 1000;
        if (perfx_trial(PERFX_JSPIN) || (getenv("SAVAGEWIND_PERFX") && (strtoul(getenv("SAVAGEWIND_PERFX"), NULL, 0) & PERFX_JSPIN)))
            g_spin_alt = getenv("SAVAGEWIND_JOBS_SPIN_ALT") ? atoi(getenv("SAVAGEWIND_JOBS_SPIN_ALT")) : 50;
        g_on = !e || atoi(e) != 0;
        LOG("[jobs] %s\n", g_on ? "native scheduler (SAVAGEWIND_JOBS_NATIVE=0: the game's own)" : "the game's own scheduler");
    }
    return g_on;
}

static Pool *pool_of(uint64_t pool) {
    for (int i = 0; i < 8; i++)
        if (__atomic_load_n(&g_pools[i].pool, __ATOMIC_ACQUIRE) == pool) return &g_pools[i];
    pthread_mutex_lock(&g_pools_m);
    Pool *p = NULL;
    for (int i = 0; i < 8 && !p; i++)
        if (g_pools[i].pool == pool) p = &g_pools[i];
    for (int i = 0; i < 8 && !p; i++)
        if (!g_pools[i].pool) {
            p = &g_pools[i];
            __atomic_store_n(&p->pool, pool, __ATOMIC_RELEASE);
            LOG("[jobs] pool %llx: %u workers\n", (unsigned long long)pool, LD32(pool + POOL_COUNT));
        }
    pthread_mutex_unlock(&g_pools_m);
    if (!p) LOG("[jobs] more than 8 pools: %llx runs on the game's scheduler\n", (unsigned long long)pool);
    return p;
}

static Pool *pool_of_worker(uint64_t w) {
    for (int i = 0; i < 8; i++) {
        uint64_t pool = __atomic_load_n(&g_pools[i].pool, __ATOMIC_ACQUIRE);
        if (!pool) continue;
        uint32_t n = LD32(pool + POOL_COUNT);
        for (uint32_t j = 0; j < n; j++)
            if (LD64(LD64(pool + POOL_WORKERS) + 8 * j) == w) return &g_pools[i];
    }
    return NULL;
}

static void bump(uint32_t *w) {
    __atomic_add_fetch(w, 1, __ATOMIC_RELEASE);
    plat_wake_u32_all(w);
}

// The workers' wake-up for new work (a round opens, a group joins one): the thread that has it - the main thread, on
// the frame's critical path - wakes one sleeping worker, and the first worker up wakes the rest (fan_out), so the
// operating system's wake calls (one per sleeping thread on Windows: NtAlertThreadByThreadId) run on a worker in
// parallel with the work instead of one after another on the main thread (~3% of its time at N = 6, 720p).
// SAVAGEWIND_JOBS_WAKE_ALL=1: every worker from the waking thread, as before; PERFX bit 32768 A/Bs it.
static int g_wake_all = -1;
static bool wake_fan(void) {
    if (g_wake_all < 0) g_wake_all = getenv("SAVAGEWIND_JOBS_WAKE_ALL") && atoi(getenv("SAVAGEWIND_JOBS_WAKE_ALL"));
    if (g_wake_all) return false;
    extern uint32_t g_perfx_mask;
    extern uint32_t perfx_now(void);
    return !(g_perfx_mask & 32768) || (perfx_now() & 32768);
}
static void wake_workers(Pool *p) {
    __atomic_add_fetch(&p->seq, 1, __ATOMIC_RELEASE);
    if (wake_fan()) plat_wake_u32_one(&p->seq); else plat_wake_u32_all(&p->seq);
}
static void fan_out(Pool *p, uint32_t s) {  // (a worker woken for seq s) the other sleepers, once for s
    uint32_t f = __atomic_load_n(&p->fan, __ATOMIC_ACQUIRE);
    if (f != s && __atomic_compare_exchange_n(&p->fan, &f, s, false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE))
        plat_wake_u32_all(&p->seq);
}

// Waits until *w != v: spins first, then the OS's address wait, in slices so a reboot doesn't wait it out. A thread
// that holds an emulated core (the main thread) lets it go while it blocks.
static void wait_change(const uint32_t *w, uint32_t v, bool core_held) {
    int spin = g_spin;
    if (g_spin_alt >= 0 && (perfx_now() & PERFX_JSPIN)) spin = g_spin_alt;
    for (int i = 0; i < spin; i++) {
        if (__atomic_load_n(w, __ATOMIC_ACQUIRE) != v) return;
        PAUSE();
    }
    if (core_held) kernel_host_work_begin();
    while (__atomic_load_n(w, __ATOMIC_ACQUIRE) == v && !g_rt_unwind) plat_wait_u32(w, v, 50000000);
    if (core_held) kernel_host_work_end();
}

static void help(Cpu *c);  // (fork-join, below)

// The worker's share of one group, as the game's run does: chunks of its grab size until the group's jobs are all
// claimed, its count of jobs added to the group's, the group's event signalled by whoever completes it; a barrier group
// (type 1 or 2) is waited for before the worker goes on.
static void run_group(Cpu *c, const uint64_t *fns, Pool *p, uint64_t w, uint32_t i, uint64_t g, uint32_t slot,
                      bool core_held) {
    ST64(w + W_GROUP, g);
    ST64(w + W_OWNER, LD64(g + G_OWNER));
    uint32_t grab = LD32(g + G_GRAB + 4 * (i < 3 ? i : 0));  // (the game's own read past three: entry 0; all 8 as made)
    bool rtr = g_rev && round_traced();
    uint64_t t0 = g_stats || g_tr_mask || rtr ? plat_now_ns() : 0;
    bool tr = g_tr_mask && trace_group(g, (uint32_t)call1(c, vfn(g, V_TOTAL), g));
    if (g_stats && slot < 64) {
        uint64_t z = 0;
        __atomic_compare_exchange_n(&g_st_first[slot], &z, t0, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
    }
    call1(c, fns[F_CLEAR], g + G_EVENT);
    uint32_t n = 0;
    uint64_t sp = c->sp;
    c->sp -= 0x10;
    for (;;) {
        c->x[0] = g, c->x[1] = grab, c->x[2] = c->sp, c->x[3] = w;
        uint32_t jt_idx = g_jt ? LD32(g + G_NEXT) : 0;  // (SAVAGEWIND_JOBS_JOBTIME: the job this call claims, as a rule)
        if (g_jt) t_nvn_tag = jobtime_is_draw(g, jt_idx);
        uint64_t jt0 = g_jt ? plat_now_ns() : 0, jtk = t_nvn_ticks, jtc = t_nvn_calls;
        uint64_t jsn = t_shadow_n, jsw = t_shadow_wait, jsh = t_shadow_hold;
        guest_call(c, vfn(g, V_RUN_CHUNK));
        t_nvn_tag = 0;
        bool last = c->x[0] & 1;
        if (g_jt && LD32(c->sp) == 1)
            jobtime_note(g, jt_idx, plat_now_ns() - jt0, t_nvn_ticks - jtk, t_nvn_calls - jtc, t_shadow_n - jsn,
                         t_shadow_wait - jsw, t_shadow_hold - jsh);
        n += LD32(c->sp);
        if (last) break;
    }
    c->sp = sp;
    uint32_t done = __atomic_add_fetch(P32(g + G_DONE), n, __ATOMIC_ACQ_REL);
    uint32_t total = (uint32_t)call1(c, vfn(g, V_TOTAL), g);
    if (g_stats) {
        uint64_t t1 = plat_now_ns();
        int b = sbucket(total);
        SADD(g_sb[b].thread, t1 - t0);
        if (n) SADD(g_sb[b].parts, 1);
        int kd = skind(g, total);
        if (kd >= 0) {
            SADD(g_sk[kd].thread, t1 - t0);
            if (n) SADD(g_sk[kd].parts, 1);
        }
        if (done >= total) {
            uint64_t span = slot < 64 ? t1 - __atomic_load_n(&g_st_first[slot], __ATOMIC_RELAXED) : 0;
            SADD(g_sb[b].groups, 1), SADD(g_sb[b].jobs, total), SADD(g_sb[b].grab, grab), SADD(g_sb[b].span, span);
            if (kd >= 0) SADD(g_sk[kd].groups, 1), SADD(g_sk[kd].jobs, total), SADD(g_sk[kd].span, span);
        }
    }
    uint64_t tr1 = tr || rtr ? plat_now_ns() : 0;
    if (done >= total) {
        call1(c, fns[F_SIGNAL], g + G_EVENT);
        plat_wake_u32_all(P32(g + G_DONE));
        bump(&p->evt);
    }
    ST32(w + W_STATE, 9);
    if (i < 3) {  // (the game's participation flag; nothing reads it, and past three the game's write lands in slot 0)
        __atomic_thread_fence(__ATOMIC_SEQ_CST);
        ST32(g + G_FLAGS + 4 * i, 0);
    }
    uint32_t type = LD32(g + G_TYPE);
    if (type == 1 || type == 2) {
        uint64_t b0 = g_stats ? plat_now_ns() : 0;
        for (uint32_t v = __atomic_load_n(P32(g + G_DONE), __ATOMIC_ACQUIRE);
             !(call1(c, vfn(g, V_IS_DONE), g) & 1) && !g_rt_unwind; v = __atomic_load_n(P32(g + G_DONE), __ATOMIC_ACQUIRE)) {
            help(c);
            if (v == __atomic_load_n(P32(g + G_DONE), __ATOMIC_ACQUIRE)) wait_change(P32(g + G_DONE), v, core_held);
        }
        if (g_stats) SADD(g_st_barrier, plat_now_ns() - b0);
    }
    if (rtr) {
        uint32_t e = __atomic_fetch_add(&g_rev_n, 1, __ATOMIC_RELAXED);
        uint64_t k = g_st_kick, t2 = plat_now_ns();
        if (e < 400000)
            g_rev[e] = (RoundEv){g, job_kind(g, total), (uint32_t)(__atomic_load_n(&g_round_no, __ATOMIC_RELAXED) - g_rt_first),
                                 slot, i, type, total, n, done >= total, k ? (t0 - k) / 1e3f : -1, k ? (tr1 - k) / 1e3f : -1,
                                 k ? (t2 - k) / 1e3f : -1};
    }
    if (tr) {
        uint64_t k = g_st_kick, t2 = plat_now_ns();
        LOG("[jobs-trace] group %llx type %u (%u jobs) worker %u: in %.1f, its jobs done %.1f, out %.1f us after the kick, "
            "%u jobs%s\n", (unsigned long long)g, type, total, i, k ? (t0 - k) / 1e3 : -1.0, k ? (tr1 - k) / 1e3 : -1.0,
            k ? (t2 - k) / 1e3 : -1.0, n, done >= total ? " (completed it)" : "");
        if (done >= total) __atomic_sub_fetch(&g_tr_left, 1, __ATOMIC_RELAXED);
    }
    ST32(w + W_STATE, 10);
    ST32(w + W_STATE, 2);
}

// ---- fork-join: work the runtime hands to the job threads while a job waits for it (rt_jobs_fork_join)
// A task set is published in a slot; the job threads that are idle (between groups, at a barrier, the main thread in
// its run-and-wait) claim its tasks and run them, and the thread that forked runs them too until none is left, then
// waits for the last ones to finish: a join by helping, so the work goes on whatever the others are doing. Slots are
// static; a helper counts itself in before it reads one and out after, and the owner frees its slot only when no
// helper is left in it.
typedef struct {
    int active;                // published, tasks may be claimed
    int helpers;               // threads in the slot right now
    uint32_t next, done;
    uint32_t n;
    void (*fn)(Cpu *c, void *arg, uint32_t i);
    void *arg;
    int used;                  // reserved by an owner
} __attribute__((aligned(64))) TaskSet;
#define MAX_TASKSETS 16
static TaskSet g_ts[MAX_TASKSETS];
static int g_ts_live;  // sets active now: idle threads look for work only when it is not 0

static void ts_run(Cpu *c, TaskSet *t) {  // (counted in) claims and runs t's tasks until none is left to claim
    for (uint32_t i; (i = __atomic_fetch_add(&t->next, 1, __ATOMIC_SEQ_CST)) < t->n;) {
        t->fn(c, t->arg, i);
        if (__atomic_fetch_add(&t->done, 1, __ATOMIC_SEQ_CST) + 1 == t->n) plat_wake_u32_all(&t->done);
    }
}
static void help(Cpu *c) {  // an idle job thread: any published tasks
    if (!__atomic_load_n(&g_ts_live, __ATOMIC_ACQUIRE)) return;
    for (int k = 0; k < MAX_TASKSETS; k++) {
        TaskSet *t = &g_ts[k];
        if (!__atomic_load_n(&t->active, __ATOMIC_ACQUIRE)) continue;
        __atomic_fetch_add(&t->helpers, 1, __ATOMIC_SEQ_CST);
        if (__atomic_load_n(&t->active, __ATOMIC_SEQ_CST) && __atomic_load_n(&t->next, __ATOMIC_SEQ_CST) < t->n) ts_run(c, t);
        __atomic_fetch_sub(&t->helpers, 1, __ATOMIC_SEQ_CST);
    }
}
static void wake_helpers(void) {
    for (int i = 0; i < 8; i++) {
        Pool *p = &g_pools[i];
        if (!__atomic_load_n(&p->pool, __ATOMIC_ACQUIRE)) continue;
        bump(&p->seq), bump(&p->evt);
    }
}

// Runs fn(c, arg, i) for i < n on the job threads and this one; returns when all have run. 0: no slot free (the
// caller runs them itself).
int rt_jobs_fork_join(Cpu *c, void (*fn)(Cpu *, void *, uint32_t), void *arg, uint32_t n) {
    TaskSet *t = NULL;
    for (int k = 0; k < MAX_TASKSETS && !t; k++) {
        int z = 0;
        if (__atomic_compare_exchange_n(&g_ts[k].used, &z, 1, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) t = &g_ts[k];
    }
    if (!t) return 0;
    t->fn = fn, t->arg = arg, t->n = n;
    __atomic_store_n(&t->next, 0, __ATOMIC_SEQ_CST), __atomic_store_n(&t->done, 0, __ATOMIC_SEQ_CST);
    __atomic_fetch_add(&g_ts_live, 1, __ATOMIC_SEQ_CST);
    __atomic_store_n(&t->active, 1, __ATOMIC_RELEASE);
    wake_helpers();
    __atomic_fetch_add(&t->helpers, 1, __ATOMIC_SEQ_CST);
    ts_run(c, t);
    __atomic_fetch_sub(&t->helpers, 1, __ATOMIC_SEQ_CST);
    for (uint32_t d; (d = __atomic_load_n(&t->done, __ATOMIC_ACQUIRE)) < n && !g_rt_unwind;) wait_change(&t->done, d, false);
    __atomic_store_n(&t->active, 0, __ATOMIC_SEQ_CST);
    __atomic_fetch_sub(&g_ts_live, 1, __ATOMIC_SEQ_CST);
    while (__atomic_load_n(&t->helpers, __ATOMIC_SEQ_CST)) PAUSE();  // (a helper between its check and its count: out at once)
    __atomic_store_n(&t->used, 0, __ATOMIC_RELEASE);
    return 1;
}

// The worker's share of the round's groups from *k on, in list order.
static void walk(Cpu *c, const uint64_t *fns, Pool *p, uint64_t w, uint32_t i, uint32_t *k, bool core_held) {
    uint64_t pool = p->pool, list = LD64(pool + POOL_LIST);
    uint32_t cap = LD32(pool + POOL_LIST_CAP);
    while (*k < __atomic_load_n(P32(pool + POOL_LIST_N), __ATOMIC_ACQUIRE) && !g_rt_unwind) {
        uint64_t g = LD64(list + 8ull * (*k < cap ? *k : 0));
        ++*k;
        if (i < 32 && LD32(g + G_MASK) >> i & 1) run_group(c, fns, p, w, i, g, *k - 1, core_held);
    }
}

static void run_begin(Cpu *c, const uint64_t *fns, uint64_t w) {
    ST32(w + W_RUNS, LD32(w + W_RUNS) + 1);
    c->x[0] = 0;
    guest_call(c, fns[F_TICK]);
    ST64(w + W_RUN_TICK, c->x[0]);
    ST32(w + W_STATE, 2);
}
static void run_end(uint64_t w) {
    ST64(w + W_GROUP, 0);
    ST64(w + W_OWNER, 0);
    ST32(w + W_STATE, 0);
}

// A job worker's thread from its first run on: an OS-scheduled thread off the emulated cores, walking each round as
// it opens and taking groups that join it; between rounds it waits on the pool's word.
static void worker_loop(Cpu *c, const uint64_t *fns, Pool *p, uint64_t w, uint32_t i) {
    kernel_thread_off_cores();
    uint32_t gen = UINT32_MAX, k = 0, gen_seen = __atomic_load_n(&p->seq, __ATOMIC_ACQUIRE);
    // (testing) SAVAGEWIND_JOBS_LOWPRIO_FROM=<i>: workers from index i at below-normal priority - background workers
    // that take jobs when a core is idle (PERFX_JLOWPRIO: only in the on blocks)
    static int low_from = -2;
    if (low_from == -2) low_from = getenv("SAVAGEWIND_JOBS_LOWPRIO_FROM") ? atoi(getenv("SAVAGEWIND_JOBS_LOWPRIO_FROM")) : -1;
    int prio_now = 0;
    while (!g_rt_unwind) {
        if (low_from >= 0 && (int)i >= low_from) {
            int want = !perfx_trial(PERFX_JLOWPRIO) || (perfx_now() & PERFX_JLOWPRIO) ? -1 : 0;
            if (want != prio_now) prio_now = want, plat_thread_priority(want);
        }
        uint32_t s = __atomic_load_n(&p->seq, __ATOMIC_ACQUIRE);
        if (s != gen_seen) gen_seen = s, fan_out(p, s);  // (new work: the workers still asleep up too)
        help(c);  // (tasks a job forked: rt_jobs_fork_join)
        if (__atomic_load_n(&p->open, __ATOMIC_SEQ_CST)) {
            __atomic_add_fetch(&p->busy, 1, __ATOMIC_SEQ_CST);
            if (__atomic_load_n(&p->open, __ATOMIC_SEQ_CST)) {  // (run-and-wait closes, then waits for busy = 0)
                uint32_t r = __atomic_load_n(&p->gen, __ATOMIC_ACQUIRE);
                if (r != gen) gen = r, k = 0;
                if (k < __atomic_load_n(P32(p->pool + POOL_LIST_N), __ATOMIC_ACQUIRE)) {
                    if (g_stats && !k) {
                        uint64_t kick = g_st_kick;
                        if (kick) SADD(g_st_wake, plat_now_ns() - kick), SADD(g_st_wakes, 1);
                    }
                    run_begin(c, fns, w);
                    walk(c, fns, p, w, i, &k, false);
                    run_end(w);
                }
            }
            if (!__atomic_sub_fetch(&p->busy, 1, __ATOMIC_SEQ_CST)) plat_wake_u32_all(&p->busy);
        }
        wait_change(&p->seq, s, false);
    }
}

// f_7000f93ea0 push(pool, owner, group, mask, type, front): the group set up and reset by the game's code, recorded
// in the round list; a round already running takes it at once.
int rt_jobs_push(Cpu *c, const uint64_t *fns) {
    uint64_t pool = c->x[0], owner = c->x[1], g = c->x[2];
    uint32_t mask = (uint32_t)c->x[3], type = (uint32_t)c->x[4];
    if (!on() || LD8(pool + POOL_SERIAL)) return 0;
    Pool *p = pool_of(pool);
    if (!p) return 0;
    // sead's "every core" mask, built for the Switch's six (an unrolled 1 << 0..5, f_7000f935c8 and others: gsys's
    // model jobs, 90 a frame): no core past five existed, so 0x3f meant all of them - here all of the pool's workers.
    // (PERFX_MASK6: the console's six in the off blocks)
    if (mask == 0x3f && !(perfx_trial(PERFX_MASK6) && !(perfx_now() & PERFX_MASK6))) {
        uint32_t count = LD32(pool + POOL_COUNT);
        mask = count >= 32 ? ~0u : (1u << count) - 1;
    }
    uint64_t push_lr = c->x[30];
    c->x[0] = g, c->x[1] = mask, c->x[2] = type;
    guest_call(c, fns[F_SETUP]);
    call1(c, vfn(g, V_RESET), g);
    {   // (research) SAVAGEWIND_JOBS_PUSH_LOG=<n>: the first n pushes - call site, the first job's class, mask, jobs
        static int left = -1;
        if (left < 0) left = getenv("SAVAGEWIND_JOBS_PUSH_LOG") ? atoi(getenv("SAVAGEWIND_JOBS_PUSH_LOG")) : 0;
        extern uint64_t g_game_frames;  // (from game frame 3000: the steady state, not the load)
        if (left > 0 && g_game_frames >= 3000 && __atomic_sub_fetch(&left, 1, __ATOMIC_RELAXED) >= 0) {
            uint32_t cap = LD32(g + 0x80), nj = LD32(g + G_JOBS);
            uint64_t arr = LD64(g + 0x88), job = cap && arr >= GUEST_LO && arr < GUEST_HI ? LD64(arr) : 0;
            uint64_t vt = job >= GUEST_LO && job < GUEST_HI ? LD64(job) : 0;
            LOG("[jobs-push] lr %llx group %llx class %llx +0x18 %u mask %#x type %u jobs %u\n", (unsigned long long)push_lr,
                (unsigned long long)g, (unsigned long long)vt, vt ? LD32(job + 0x18) : 0, mask, type, nj);
        }
    }
    uint32_t count = LD32(pool + POOL_COUNT);
    if (mask & (count >= 32 ? ~0u : (1u << count) - 1)) ST64(g + G_OWNER, owner);  // (the game: in each ring push)
    uint32_t n = LD32(pool + POOL_LIST_N), cap = LD32(pool + POOL_LIST_CAP);
    ST64(LD64(pool + POOL_LIST) + 8ull * (n < cap ? n : 0), g);
    __atomic_store_n(P32(pool + POOL_LIST_N), n + 1, __ATOMIC_RELEASE);
    if (__atomic_load_n(&p->open, __ATOMIC_SEQ_CST)) wake_workers(p), bump(&p->evt);
    return 1;
}

// Opens the pool's round: its workers' states as the game's kick sets them, and a worker's thread started into the
// native loop by the game's thread message the first time.
static void open_round(Cpu *c, Pool *p) {
    uint64_t pool = p->pool;
    uint32_t count = LD32(pool + POOL_COUNT), n = LD32(pool + POOL_LIST_N), cap = LD32(pool + POOL_LIST_CAP), masks = 0;
    for (uint32_t k = 0; k < n; k++) masks |= LD32(LD64(LD64(pool + POOL_LIST) + 8ull * (k < cap ? k : 0)) + G_MASK);
    for (uint32_t i = 1; i < count && i < 32; i++) {
        uint64_t w = LD64(LD64(pool + POOL_WORKERS) + 8ull * i);
        if (masks >> i & 1) ST32(w + W_STATE, 1);
        if (!p->started[i]) {
            p->started[i] = 1;
            c->x[0] = w, c->x[1] = 1, c->x[2] = 1;
            guest_call(c, vfn(w, V_SEND));
        }
    }
    if (g_stats || g_tr_mask || g_rev) g_st_kick = plat_now_ns();
    __atomic_store_n(&p->open, 1, __ATOMIC_SEQ_CST);
    wake_workers(p);
}

// f_7000f93f78 kick(pool)
int rt_jobs_kick(Cpu *c, const uint64_t *fns) {
    uint64_t pool = c->x[0];
    if (!on() || LD8(pool + POOL_SERIAL)) return 0;
    Pool *p = pool_of(pool);
    if (!p) return 0;
    ST32(pool + POOL_KICKS, LD32(pool + POOL_KICKS) + 1);
    c->x[0] = 0;
    guest_call(c, fns[F_TICK]);
    ST64(pool + POOL_KICK_TICK, c->x[0]);
    open_round(c, p);
    return 1;
}

// f_7000f94580 run(worker): on a job worker's own thread (its first message), the native loop for good. Index 0's
// run in place is only the game's serial path.
int rt_jobs_run(Cpu *c, const uint64_t *fns) {
    uint64_t w = c->x[0];
    uint32_t i = LD32(w + W_INDEX);
    if (!on() || i == 0) return 0;
    Pool *p = pool_of_worker(w);
    if (!p) return 0;
    LOG("[jobs] worker %u on its own thread, off the emulated cores\n", i);
    worker_loop(c, fns, p, w, i);
    return 1;
}

// A broadcast to every core (f_700127d4d0, BOTW-JOBS-NATIVE.md #30): the game makes one job group per core on its stack
// - three, for sead's three cores - and pushes each with its core's bit, then runs the round and finishes them. Past
// three the groups are here, in a per-core block of the function (the block's first three slots unused), made as the
// game makes its own: the constructor, the stack groups' class, grab size 1; and finished after the round as the game
// finishes its own (the base class back, the group's event finalised). fns: the group constructor, set-grab-size, the
// event's finaliser, the GOT entry of the base class's vtable and the patched loop's address (the block's key). (The
// function runs on the main thread, one at a time.)
extern uint64_t rt_percore_block_live(uint64_t obj, uint32_t field, uint32_t stride, uint32_t switch_slots);
enum { BC_CTOR, BC_GRAB, BC_FINI, BC_BASE_VT, BC_KEY };
static uint32_t g_bcast_n;  // groups made past three in the current broadcast

uint64_t rt_jobs_core_group(Cpu *c, const uint64_t *fns, uint64_t first, uint32_t k) {
    uint64_t g = rt_percore_block_live(fns[BC_KEY], 0, 0x80, 3) + 0x80ull * k;
    call1(c, fns[BC_CTOR], g);
    ST64(g, LD64(first));  // (the stack groups' class)
    c->x[0] = g, c->x[1] = 1;
    guest_call(c, fns[BC_GRAB]);
    if (k + 1 > g_bcast_n + 3) g_bcast_n = k - 2;
    return g;
}

void rt_jobs_core_groups_fini(Cpu *c, const uint64_t *fns) {
    uint64_t block = g_bcast_n ? rt_percore_block_live(fns[BC_KEY], 0, 0x80, 3) : 0;
    for (uint32_t k = 3; k < 3 + g_bcast_n; k++) {
        uint64_t g = block + 0x80ull * k;
        ST64(g, LD64(fns[BC_BASE_VT]) + 0x10);
        call1(c, fns[BC_FINI], g + G_EVENT);
    }
    g_bcast_n = 0;
}

// The parts the views' recording cuts each draw list into (tools/patches.py _view_rec_parts; the Switch build: 3), each
// a contiguous range recorded into its own command buffer by whichever core takes it: one per game core, the way a PC
// engine records a pass's draws in parallel. SAVAGEWIND_VIEW_REC_PARTS=<n> (research): another count.
uint32_t rt_view_rec_parts(void) {
    extern int rt_game_cores(void);
    static int n;
    if (n) return (uint32_t)n;
    const char *e = getenv("SAVAGEWIND_VIEW_REC_PARTS");
    n = e && atoi(e) > 0 ? atoi(e) : rt_game_cores();
    LOG("[jobs] view recording: draw lists in up to %d parts\n", n);
    return (uint32_t)n;
}

// (testing) SAVAGEWIND_VIEW_REC_AB=1: the views' lists cut into the Switch build's 3 parts in odd game frames and into
// the cores' count in even ones (tools/patches.py _view_rec_cut, where f_7000fb98bc reads the count each cut), for
// consecutive frames compared in one run.
uint32_t rt_view_rec_cut(uint32_t stored) {
    static int ab = -1;
    if (ab < 0) ab = getenv("SAVAGEWIND_VIEW_REC_AB") && atoi(getenv("SAVAGEWIND_VIEW_REC_AB"));
    extern uint64_t g_game_frames;
    if (perfx_trial(PERFX_VRPARTS) && !(perfx_now() & PERFX_VRPARTS)) return 3;  // (in-launch: 3 parts in the off blocks)
    return ab && (g_game_frames & 1) ? 3 : stored;
}

// A draw list no view cut into parts (tools/patches.py _view_rec_inline): the consumer f_7000fab5e0(list, draw
// context, pass, view, x4) found no record of parts for the list's pass and is about to record the whole list on the
// one thread drawing the layer. Instead its parts - contiguous ranges, at least VRI_MIN draws each, up to the view parts'
// count - are recorded on every core by the game's part recorder, each in a command buffer of its own as the views'
// parts are, and the consumer is entered again: it finds the record now, sorts the parts by their start and calls them
// in order (the GPU sees the list's draws in the list's order). The record is dropped after, as the game's own inline
// recording leaves none. A list another thread is doing the same for (two layers drawing it at once) is recorded the
// game's way. fns: the consumer, the part recorder f_7000fab13c, the draw context's constructor / destructor, the step
// the game's recorder takes after each part, the list's record lock / unlock.
enum { VRI_CONSUME, VRI_PART, VRI_CTX, VRI_CTX_FINI, VRI_ITEM_END, VRI_LOCK, VRI_UNLOCK, VRI_RECORD, VRI_BEGIN, VRI_END, VRI_REC };
#define VRI_MIN 8
uint64_t g_cmd_arena;  // (runtime/hle/percore.c rt_arena_setup) the command arena the parts are recorded from
typedef struct { const uint64_t *fns; uint64_t list, pass, view, x4, dctx; uint32_t total, parts; } VriTask;
static __thread int t_vri;     // (in the consumer entered again: its search failing again records the game's way)
static uint64_t g_vri_lists[16];
static void vri_part(Cpu *c, void *arg, uint32_t i) {
    VriTask *t = arg;
    uint32_t per = t->total / t->parts, start = i * per, count = i + 1 == t->parts ? t->total - start : per;
    uint64_t sp = c->sp;
    c->sp = (sp - 0x200) & ~15ull;
    uint64_t ctx = c->sp, range = ctx + 0x1d0;
    ST32(range, start), ST32(range + 4, count);
    c->x[0] = ctx;
    guest_call(c, t->fns[VRI_CTX]);
    // (the layer's settings: its context's graphics state +8 up to the command buffer's +0xb8, and +0xf9; the part's
    // own command buffer fields stay as made - +0xb8, +0xe8, +0xf8 and the wrapper at +0x100)
    memcpy((void *)(uintptr_t)(ctx + 8), (const void *)(uintptr_t)(t->dctx + 8), 0xb0);
    memcpy((void *)(uintptr_t)(ctx + 0xc0), (const void *)(uintptr_t)(t->dctx + 0xc0), 0x28);
    memcpy((void *)(uintptr_t)(ctx + 0x1a0), (const void *)(uintptr_t)(t->dctx + 0x1a0), 0x30);
    ST8(ctx + 0xf9, LD8(t->dctx + 0xf9));
    // (f_7000fab13c's steps with the inline recording's mode: the list's record for the pass, a command buffer begun
    // with the part's start as its key, the range recorded inheriting the state the layer set where it is called - the
    // views' part mode would set the pass's defaults again over it - the buffer ended and added to the record)
    c->x[0] = t->list, c->x[1] = t->pass;
    guest_call(c, t->fns[VRI_RECORD]);
    uint64_t rec = c->x[0];
    c->x[0] = g_cmd_arena, c->x[1] = ctx, c->x[2] = rec + 0x38, c->x[3] = start;
    guest_call(c, t->fns[VRI_BEGIN]);
    uint32_t key = (uint32_t)c->x[0];
    c->x[0] = t->list, c->x[1] = ctx, c->x[2] = rec, c->x[3] = t->view, c->x[4] = range, c->x[5] = 0, c->x[6] = t->x4;
    guest_call(c, t->fns[VRI_REC]);  // (x6 as the consumer passes it: its caller's view data, not a scratch word)
    c->x[0] = g_cmd_arena, c->x[1] = ctx, c->x[2] = key;
    guest_call(c, t->fns[VRI_END]);
    uint64_t dl = c->x[0];
    if (LD8(dl + 0x24) & 1) {
        uint32_t k = __atomic_fetch_add(P32(rec + 0x58), 1, __ATOMIC_ACQ_REL);
        ST64(LD64(rec + 0x60) + 8ull * k, dl);
    }
    guest_call(c, t->fns[VRI_ITEM_END]);
    c->x[0] = ctx;
    guest_call(c, t->fns[VRI_CTX_FINI]);
    c->sp = sp;
}
int rt_view_rec_inline(Cpu *c, const uint64_t *fns) {
    // (off unless SAVAGEWIND_VIEW_REC_INLINE=1: at Kakariko characters and foliage come out unlit - the material lists
    // the lighting reads - with a new draw context, with the layer's context settings copied in as below too, and with
    // the consumer's x6. Not the context then; open: what else the inline recording depends on - being recorded into the
    // layer's own command buffer at that point. The views' part mode was worse: the pass defaults over the layer's
    // per-tile state)
    static int enabled = -1;
    if (enabled < 0) enabled = getenv("SAVAGEWIND_VIEW_REC_INLINE") && atoi(getenv("SAVAGEWIND_VIEW_REC_INLINE"));
    if (!enabled) return 0;
    uint64_t list = c->x[23], dctx = c->x[19], pass = c->x[22], view = c->x[21], x4 = c->x[20];
    uint32_t total = LD32(list + 0x3c), parts = rt_view_rec_parts();
    if (total / VRI_MIN < parts) parts = total / VRI_MIN;
    if (t_vri || parts < 2 || !on() || !g_cmd_arena) return 0;
    {   // (SAVAGEWIND_PERFX=16384 with _AB: the game's inline recording in the "off" blocks, an A/B in one launch)
        extern uint32_t g_perfx_mask;
        extern uint32_t perfx_now(void);
        if ((g_perfx_mask & 16384) && !(perfx_now() & 16384)) return 0;
    }
    {   // (the parts come from the arena the layer's own command buffer is from)
        uint64_t cb = LD64(dctx + 0xe8), arr = LD64(g_cmd_arena + 0x108);
        if (cb < arr || cb >= arr + 0x288ull * LD32(g_cmd_arena + 0x100)) {
            static int told;
            if (!told++) LOG("[jobs] view recording: a draw context's command buffer %llx not from the arena: lists recorded "
                             "inline\n", (unsigned long long)cb);
            return 0;
        }
    }
    int slot = -1;
    for (int k = 0; k < 16 && slot < 0; k++) {
        uint64_t z = 0;
        if (__atomic_load_n(&g_vri_lists[k], __ATOMIC_SEQ_CST) == list) return 0;
        if (__atomic_compare_exchange_n(&g_vri_lists[k], &z, list, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) slot = k;
    }
    if (slot < 0) return 0;
    for (int k = 0; k < 16; k++)  // (another thread took the same list in the meantime)
        if (k != slot && __atomic_load_n(&g_vri_lists[k], __ATOMIC_SEQ_CST) == list) { __atomic_store_n(&g_vri_lists[slot], 0, __ATOMIC_SEQ_CST); return 0; }
    uint32_t n0 = LD32(list + 0x50);
    VriTask t = {fns, list, pass, view, x4, dctx, total, parts};
    if (!rt_jobs_fork_join(c, vri_part, &t, parts))
        for (uint32_t i = 0; i < parts; i++) vri_part(c, &t, i);
    t_vri = 1;
    c->x[0] = list, c->x[1] = dctx, c->x[2] = pass, c->x[3] = view, c->x[4] = x4;
    guest_call(c, fns[VRI_CONSUME]);
    t_vri = 0;
    c->x[0] = list + 0x58;
    guest_call(c, fns[VRI_LOCK]);
    if (LD32(list + 0x50) == n0 + 1) ST32(list + 0x50, n0);
    c->x[0] = list + 0x58;
    guest_call(c, fns[VRI_UNLOCK]);
    __atomic_store_n(&g_vri_lists[slot], 0, __ATOMIC_SEQ_CST);
    static int logged;
    if (__atomic_fetch_add(&logged, 1, __ATOMIC_RELAXED) < 4)
        LOG("[jobs] view recording: list %llx (%u draws) recorded in %u parts on every core\n", (unsigned long long)list,
            total, parts);
    return 1;
}

// (research) SAVAGEWIND_LAYER_STEP_DL=1: the layers' draws keep agl's one display list per render step (tools/patches.py
// _layer_step_dl), what running a layer's steps on every core builds on; SAVAGEWIND_LAYER_STEPS=1 then does that.
int rt_layer_step_dl(void) {
    const char *e = getenv("SAVAGEWIND_LAYER_STEP_DL");
    return e && atoi(e);
}

// (research, off: SAVAGEWIND_LAYER_STEPS=1 with SAVAGEWIND_LAYER_STEP_DL=1) Not right yet: a step's draw methods use per-core
// entries the layer's setup fills only for the core its job runs on (f_70010642a8: [sp+0x60] + core x 0xa00, the
// pointer from f_7000ed6364 at +0x4f8 null on another core: a crash at Kakariko); the step display lists alone are fine.
// A layer's render steps with a display list each (f_7000ef4a5c, tools/patches.py _layer_steps; registers as at its
// loop head: x22 the renderer, x24 the job's draw context, x23 the layer, w21 the flags, w20 a list per step, w25 the
// steps, x19 the begin's first word, [sp + 0xc] the steps' first key). The steps to draw are found as the loop finds
// them - the layer's own tests, +0x78 (skip) unless flags bit 0, +0x80 (which pass: 4 / 2, against the flags) - and
// recorded on every core: each with a draw context of its own and the step info made for it as the loop makes it, its
// display list begun with key first + i and put in the renderer's list, which the submit sorts by key - the GPU gets
// the steps in order, as from one thread. 0: the loop does it.
enum { LS_INFO, LS_DRAW, LS_INSERT, LS_BEGIN, LS_END, LS_CTX, LS_CTX_FINI, LS_DLG_GOT };
typedef struct { const uint64_t *fns; uint64_t r, layer, dlg0, dlg1; uint32_t key0, n, steps[64], turn; } LsTask;
static pthread_mutex_t g_ls_one = PTHREAD_MUTEX_INITIALIZER;  // (SAVAGEWIND_LAYER_STEPS=2: one step at a time)
static int g_ls_mode;
static void ls_step(Cpu *c, void *arg, uint32_t k) {
    LsTask *t = arg;
    if (g_ls_mode == 2) pthread_mutex_lock(&g_ls_one);
    if (g_ls_mode == 4)  // (testing: other threads, the steps' order kept)
        for (uint32_t v; (v = __atomic_load_n(&t->turn, __ATOMIC_ACQUIRE)) != k;) wait_change(&t->turn, v, false);
    const uint64_t *fns = t->fns;
    uint32_t i = t->steps[k];
    uint64_t sp = c->sp, arena = LD64(t->r + 0x7d8);
    c->sp = (sp - 0x300) & ~15ull;
    uint64_t ctx = c->sp, info = ctx + 0x1e0, dlg = ctx + 0x240;
    c->x[0] = ctx;
    guest_call(c, fns[LS_CTX]);
    c->x[0] = info, c->x[1] = ctx, c->x[2] = LD8(t->r + 8), c->x[3] = 0, c->x[4] = LD64(t->r + 0x10),
    c->x[5] = (LD32(t->r + 0x838) >> 3) & 1, c->x[6] = t->layer;
    guest_call(c, fns[LS_INFO]);
    ST32(info, i);
    ST64(dlg, t->dlg0), ST64(dlg + 8, t->dlg1);
    c->x[0] = arena, c->x[1] = ctx, c->x[2] = dlg, c->x[3] = t->key0 + i;
    guest_call(c, fns[LS_BEGIN]);
    uint32_t key = (uint32_t)c->x[0];
    c->x[0] = t->layer, c->x[1] = info;
    guest_call(c, fns[LS_DRAW]);
    if (key != UINT32_MAX) {
        c->x[0] = arena, c->x[1] = ctx, c->x[2] = key;
        guest_call(c, fns[LS_END]);
        uint64_t dl = c->x[0];
        c->x[0] = t->r, c->x[1] = ctx, c->x[2] = dl, c->x[3] = t->layer;
        guest_call(c, fns[LS_INSERT]);
    }
    c->x[0] = ctx;
    guest_call(c, fns[LS_CTX_FINI]);
    c->sp = sp;
    if (g_ls_mode == 2) pthread_mutex_unlock(&g_ls_one);
    if (g_ls_mode == 4) __atomic_store_n(&t->turn, k + 1, __ATOMIC_RELEASE), plat_wake_u32_all(&t->turn);
}
int rt_layer_steps(Cpu *c, const uint64_t *fns) {
    static int enabled = -1;
    if (enabled < 0) enabled = getenv("SAVAGEWIND_LAYER_STEPS") ? atoi(getenv("SAVAGEWIND_LAYER_STEPS")) : 0, g_ls_mode = enabled;
    uint32_t count = (uint32_t)c->x[25], flags = (uint32_t)c->x[21];
    if (!enabled || !(c->x[20] & 1) || count < 2 || count > 64 || !on()) return 0;
    LsTask t = {fns, c->x[22], c->x[23], c->x[19], LD64(LD64(fns[LS_DLG_GOT]) + 8), LD32(c->sp + 0xc), 0, {0}, 0};
    for (uint32_t i = 0; i < count; i++) {
        uint64_t vt = LD64(t.layer);
        bool skip = false;
        if (!(flags & 1)) {
            c->x[0] = t.layer, c->x[1] = i;
            guest_call(c, LD64(vt + 0x78));
            skip = c->x[0] & 1;
        }
        c->x[0] = t.layer, c->x[1] = i;
        guest_call(c, LD64(vt + 0x80));
        if (!skip && ((c->x[0] & 1 ? 4u : 2u) & flags)) t.steps[t.n++] = i;
    }
    if (g_ls_mode == 5 && t.n > 2) {  // (testing: the first step here, then the rest at once on every core)
        ls_step(c, &t, 0);
        LsTask r = t;
        r.n = t.n - 1;
        memmove(r.steps, t.steps + 1, r.n * sizeof r.steps[0]);
        if (!rt_jobs_fork_join(c, ls_step, &r, r.n))
            for (uint32_t k = 0; k < r.n; k++) ls_step(c, &r, k);
        return 1;
    }
    if (g_ls_mode == 3) {  // (testing: every step on this thread, in order, each with a context of its own)
        for (uint32_t k = 0; k < t.n; k++) ls_step(c, &t, k);
        return 1;
    }
    if (t.n > 1 && rt_jobs_fork_join(c, ls_step, &t, t.n)) return 1;
    for (uint32_t k = 0; k < t.n; k++) ls_step(c, &t, k);
    return 1;
}

// Whether sead thread id `id` is one of the job system's threads (BOTW-JOBS-NATIVE.md #31): the main thread's (it runs
// index 0's share of each round; worker 0's own thread is never started) or a worker's of the pool. BOTW asks it in
// BaseProcMgr's "high priority thread" check and in ResourceMgrTask's load request, each time against a copy of the set
// taken at boot for three cores (main, workers 1-2; main, workers 0-2): here the pool answers, for every worker it has
// (tools/patches.py _job_thread_proc / _job_thread_res). Called on each proc fetch: a walk of the pool's workers.
int rt_jobs_job_thread(uint64_t pool, uint32_t main_id, uint32_t id) {
    if (id == main_id) return 1;
    if (!pool) return 0;
    uint32_t n = LD32(pool + POOL_COUNT);
    uint64_t ws = LD64(pool + POOL_WORKERS);
    static int logged;
    if (!logged && !__atomic_exchange_n(&logged, 1, __ATOMIC_RELAXED)) {
        LOG("[jobs] job system threads: main %u, %u workers:", main_id, n);
        for (uint32_t i = 0; i < n; i++) LOG(" %u", LD64(ws + 8ull * i) ? LD32(LD64(ws + 8ull * i) + W_ID) : 0);
        LOG("\n");
    }
    for (uint32_t i = 1; i < n; i++) {  // (worker 0's thread never runs: its id is 0)
        uint64_t w = LD64(ws + 8ull * i);  // (0 while the pool is being made)
        if (w && LD32(w + W_ID) == id) return 1;
    }
    return 0;
}

static void check_round(Cpu *c, uint64_t pool, uint32_t n) {
    uint32_t cap = LD32(pool + POOL_LIST_CAP);
    for (uint32_t k = 0; k < n; k++) {
        uint64_t g = LD64(LD64(pool + POOL_LIST) + 8ull * (k < cap ? k : 0));
        uint32_t total = (uint32_t)call1(c, vfn(g, V_TOTAL), g), done = LD32(g + G_DONE);
        bool counted = total == LD32(g + G_JOBS);  // (the base group's count and claim counter; a subclass has its own)
        if (done != total || (counted && LD32(g + G_NEXT) < total)) {
            g_bad++;
            LOG("[jobs] check: group %llx (mask %x) done %u of %u, claimed %u of %u\n", (unsigned long long)g,
                LD32(g + G_MASK), done, total, LD32(g + G_NEXT), LD32(g + G_JOBS));
        }
    }
    g_groups += n;
    if (!(++g_rounds & 4095))
        LOG("[jobs] check: %llu rounds, %llu groups, %llu wrong\n", (unsigned long long)g_rounds,
            (unsigned long long)g_groups, (unsigned long long)g_bad);
}

// (SAVAGEWIND_JOBS_STATS) at a round's end on the main thread: t its run-and-wait's start, run its time running groups
static void stats_round(uint32_t n, uint64_t t, uint64_t run) {
    uint64_t now = plat_now_ns();
    g_st_rounds++;
    g_st_main_run += run;
    g_st_main_wait += now - t - run;
    if (g_st_kick && g_st_kick <= t) g_st_span += now - g_st_kick;
    g_st_kick = 0;
    for (uint32_t k = 0; k < n && k < 64; k++) __atomic_store_n(&g_st_first[k], 0, __ATOMIC_RELAXED);
    if (!g_st_t0) g_st_t0 = now;
    if (now - g_st_t0 < 2000000000ull) return;
    double s = (now - g_st_t0) / 1e9, ms = 1e6 * s;
    LOG("[jobs] stats/s: %.0f rounds; main thread runs %.1f ms, waits %.1f ms; kick to end %.1f ms; barrier waits %.1f ms\n",
        g_st_rounds / s, g_st_main_run / ms, g_st_main_wait / ms, g_st_span / ms, __atomic_load_n(&g_st_barrier, __ATOMIC_RELAXED) / ms);
    static const char *names[SB] = {"1", "2-4", "5-8", "9-16", "17-48", "49+"};
    for (int b = 0; b < SB; b++) {
        uint64_t gs = __atomic_load_n(&g_sb[b].groups, __ATOMIC_RELAXED);
        if (!gs) continue;
        LOG("[jobs]   jobs %-5s: %6.0f groups, %4.1f jobs, chunk %4.1f, threads' time %6.1f ms, first claim to done %6.1f us, "
            "threads %.2f\n", names[b], gs / s, (double)g_sb[b].jobs / gs, (double)g_sb[b].grab / gs, g_sb[b].thread / ms,
            g_sb[b].span / 1e3 / gs, (double)g_sb[b].parts / gs);
    }
    uint64_t wk = __atomic_load_n(&g_st_wakes, __ATOMIC_RELAXED);
    LOG("[jobs]   kick to a worker's start: %.1f us (%.0f a second)\n", wk ? g_st_wake / 1e3 / wk : 0.0, wk / s);
    bool shown[SK] = {0};
    for (int i = 0; i < 16; i++) {
        int h = -1;
        for (int j = 0; j < SK; j++)
            if (!shown[j] && g_sk[j].groups && (h < 0 || g_sk[j].thread > g_sk[h].thread)) h = j;
        if (h < 0) break;
        shown[h] = true;
        uint64_t gs = g_sk[h].groups;
        LOG("[jobs]   kind %010llx/%010llx type %u mask %2d workers: %6.0f /s, %5.1f jobs, threads' time %6.1f ms, "
            "first claim to done %6.1f us, threads %.2f\n", (unsigned long long)(g_sk[h].key >> 24 & 0xffffffffffull),
            (unsigned long long)(g_sk[h].key & 0xffffffffffull), g_sk[h].type, __builtin_popcount(g_sk[h].mask), gs / s,
            (double)g_sk[h].jobs / gs, g_sk[h].thread / ms, g_sk[h].span / 1e3 / gs, (double)g_sk[h].parts / gs);
    }
    memset(g_sk, 0, sizeof g_sk);
    g_st_wake = g_st_wakes = 0;
    memset(g_sb, 0, sizeof g_sb);
    g_st_rounds = g_st_main_run = g_st_main_wait = g_st_span = 0;
    __atomic_store_n(&g_st_barrier, 0, __ATOMIC_RELAXED);
    g_st_t0 = now;
}

// f_7000f94100 run-and-wait(pool), on the main thread: its own share (index 0), then every group of the round that
// some worker can run finished, the workers out of the list, and the round closed. A round nobody kicked is opened
// here (the game would leave the workers' groups in their rings for the next kick).
int rt_jobs_run_wait(Cpu *c, const uint64_t *fns) {
    uint64_t pool = c->x[0];
    if (!on() || LD8(pool + POOL_SERIAL)) return 0;
    Pool *p = pool_of(pool);
    if (!p) return 0;
    if (!__atomic_load_n(&p->open, __ATOMIC_SEQ_CST) && LD32(pool + POOL_LIST_N)) open_round(c, p);
    uint64_t w0 = LD64(LD64(pool + POOL_WORKERS));
    uint32_t count = LD32(pool + POOL_COUNT), valid = count >= 32 ? ~0u : (1u << count) - 1, k = 0, j = 0;
    run_begin(c, fns, w0);
    uint64_t st_run = 0, st_t = g_stats || g_rev ? plat_now_ns() : 0;
    for (;;) {
        uint32_t e = __atomic_load_n(&p->evt, __ATOMIC_ACQUIRE);
        if (g_stats) {
            uint64_t t = plat_now_ns();
            walk(c, fns, p, w0, 0, &k, true);
            st_run += plat_now_ns() - t;
        } else {
            walk(c, fns, p, w0, 0, &k, true);
        }
        uint32_t n = __atomic_load_n(P32(pool + POOL_LIST_N), __ATOMIC_ACQUIRE), cap = LD32(pool + POOL_LIST_CAP);
        for (; j < n; j++) {
            uint64_t g = LD64(LD64(pool + POOL_LIST) + 8ull * (j < cap ? j : 0));
            if ((LD32(g + G_MASK) & valid) && __atomic_load_n(P32(g + G_DONE), __ATOMIC_ACQUIRE) < (uint32_t)call1(c, vfn(g, V_TOTAL), g))
                break;
        }
        if ((j == n && k == n) || g_rt_unwind) break;
        help(c);
        if (k == __atomic_load_n(P32(pool + POOL_LIST_N), __ATOMIC_ACQUIRE) && e == __atomic_load_n(&p->evt, __ATOMIC_ACQUIRE))
            wait_change(&p->evt, e, true);
    }
    run_end(w0);
    __atomic_store_n(&p->open, 0, __ATOMIC_SEQ_CST);
    for (uint32_t b; (b = __atomic_load_n(&p->busy, __ATOMIC_SEQ_CST)) && !g_rt_unwind;) wait_change(&p->busy, b, true);
    uint32_t n = LD32(pool + POOL_LIST_N);
    if (g_check) check_round(c, pool, n);
    if (g_stats) stats_round(n, st_t, st_run);
    if (g_rev) {
        uint64_t r = g_round_no, k = g_st_kick, now = plat_now_ns();
        if (r >= g_rt_first && r < g_rt_first + g_rt_count) {
            extern uint64_t renderer_present_count(void);
            g_rrec[g_rrec_n++] = (RoundRec){renderer_present_count(), pool, (uint32_t)(r - g_rt_first), n,
                                            k && st_t ? (st_t - k) / 1e3f : -1, k ? (now - k) / 1e3f : -1};
        }
        __atomic_add_fetch(&g_round_no, 1, __ATOMIC_RELAXED);
        if (r + 1 == g_rt_first + g_rt_count) {  // (the last traced round ended: everything written now)
            for (uint32_t i = 0; i < g_rrec_n; i++)
                LOG("[jobs-round] R %u frame %llu pool %llx groups %u run-and-wait at %.1f end %.1f\n", g_rrec[i].round,
                    (unsigned long long)g_rrec[i].frame, (unsigned long long)g_rrec[i].pool, g_rrec[i].ngroups,
                    g_rrec[i].wait_in, g_rrec[i].end);
            uint32_t ne = g_rev_n < 400000 ? g_rev_n : 400000;
            for (uint32_t i = 0; i < ne; i++) {
                RoundEv *v = &g_rev[i];
                LOG("[jobs-round] E %u slot %u group %llx kind %llx type %u total %u w%u in %.1f jd %.1f out %.1f jobs %u%s\n",
                    v->round, v->slot, (unsigned long long)v->g, (unsigned long long)v->kind, v->type, v->total, v->worker,
                    v->in, v->jd, v->out, v->jobs, v->done ? " done" : "");
            }
            g_rev = NULL;
        }
    }
    __atomic_store_n(P32(pool + POOL_LIST_N), 0, __ATOMIC_RELEASE);
    __atomic_add_fetch(&p->gen, 1, __ATOMIC_RELEASE);
    return 1;
}
