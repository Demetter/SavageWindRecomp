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
    W_INDEX = 0xfc, W_STATE = 0x100, W_GROUP = 0x130, W_OWNER = 0x138, W_RUNS = 0x140, W_RUN_TICK = 0x148,
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
    uint8_t started[32];  // worker i's thread is in the native loop
} Pool;
static Pool g_pools[8];
static pthread_mutex_t g_pools_m = PTHREAD_MUTEX_INITIALIZER;
static int g_on = -1, g_check, g_spin;
static uint64_t g_rounds, g_groups, g_bad;

RT_ON_RESET(jobs_reboot_reset) { memset(g_pools, 0, sizeof g_pools); }  // (reboot: pools and threads were the guest's)

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
        g_spin = s ? atoi(s) : 1000;
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

// Waits until *w != v: spins first, then the OS's address wait, in slices so a reboot doesn't wait it out. A thread
// that holds an emulated core (the main thread) lets it go while it blocks.
static void wait_change(const uint32_t *w, uint32_t v, bool core_held) {
    for (int i = 0; i < g_spin; i++) {
        if (__atomic_load_n(w, __ATOMIC_ACQUIRE) != v) return;
        PAUSE();
    }
    if (core_held) kernel_host_work_begin();
    while (__atomic_load_n(w, __ATOMIC_ACQUIRE) == v && !g_rt_unwind) plat_wait_u32(w, v, 50000000);
    if (core_held) kernel_host_work_end();
}

// The worker's share of one group, as the game's run does: chunks of its grab size until the group's jobs are all
// claimed, its count of jobs added to the group's, the group's event signalled by whoever completes it; a barrier group
// (type 1 or 2) is waited for before the worker goes on.
static void run_group(Cpu *c, const uint64_t *fns, Pool *p, uint64_t w, uint32_t i, uint64_t g, bool core_held) {
    ST64(w + W_GROUP, g);
    ST64(w + W_OWNER, LD64(g + G_OWNER));
    uint32_t grab = LD32(g + G_GRAB + 4 * (i < 3 ? i : 0));  // (the game's own read past three: entry 0; all 8 as made)
    call1(c, fns[F_CLEAR], g + G_EVENT);
    uint32_t n = 0;
    uint64_t sp = c->sp;
    c->sp -= 0x10;
    for (;;) {
        c->x[0] = g, c->x[1] = grab, c->x[2] = c->sp, c->x[3] = w;
        guest_call(c, vfn(g, V_RUN_CHUNK));
        bool last = c->x[0] & 1;
        n += LD32(c->sp);
        if (last) break;
    }
    c->sp = sp;
    uint32_t done = __atomic_add_fetch(P32(g + G_DONE), n, __ATOMIC_ACQ_REL);
    if (done >= (uint32_t)call1(c, vfn(g, V_TOTAL), g)) {
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
    if (type == 1 || type == 2)
        for (uint32_t v = __atomic_load_n(P32(g + G_DONE), __ATOMIC_ACQUIRE);
             !(call1(c, vfn(g, V_IS_DONE), g) & 1) && !g_rt_unwind; v = __atomic_load_n(P32(g + G_DONE), __ATOMIC_ACQUIRE))
            wait_change(P32(g + G_DONE), v, core_held);
    ST32(w + W_STATE, 10);
    ST32(w + W_STATE, 2);
}

// The worker's share of the round's groups from *k on, in list order.
static void walk(Cpu *c, const uint64_t *fns, Pool *p, uint64_t w, uint32_t i, uint32_t *k, bool core_held) {
    uint64_t pool = p->pool, list = LD64(pool + POOL_LIST);
    uint32_t cap = LD32(pool + POOL_LIST_CAP);
    while (*k < __atomic_load_n(P32(pool + POOL_LIST_N), __ATOMIC_ACQUIRE) && !g_rt_unwind) {
        uint64_t g = LD64(list + 8ull * (*k < cap ? *k : 0));
        ++*k;
        if (i < 32 && LD32(g + G_MASK) >> i & 1) run_group(c, fns, p, w, i, g, core_held);
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
    uint32_t gen = UINT32_MAX, k = 0;
    while (!g_rt_unwind) {
        uint32_t s = __atomic_load_n(&p->seq, __ATOMIC_ACQUIRE);
        if (__atomic_load_n(&p->open, __ATOMIC_SEQ_CST)) {
            __atomic_add_fetch(&p->busy, 1, __ATOMIC_SEQ_CST);
            if (__atomic_load_n(&p->open, __ATOMIC_SEQ_CST)) {  // (run-and-wait closes, then waits for busy = 0)
                uint32_t r = __atomic_load_n(&p->gen, __ATOMIC_ACQUIRE);
                if (r != gen) gen = r, k = 0;
                if (k < __atomic_load_n(P32(p->pool + POOL_LIST_N), __ATOMIC_ACQUIRE)) {
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
    c->x[0] = g, c->x[1] = mask, c->x[2] = type;
    guest_call(c, fns[F_SETUP]);
    call1(c, vfn(g, V_RESET), g);
    uint32_t count = LD32(pool + POOL_COUNT);
    if (mask & (count >= 32 ? ~0u : (1u << count) - 1)) ST64(g + G_OWNER, owner);  // (the game: in each ring push)
    uint32_t n = LD32(pool + POOL_LIST_N), cap = LD32(pool + POOL_LIST_CAP);
    ST64(LD64(pool + POOL_LIST) + 8ull * (n < cap ? n : 0), g);
    __atomic_store_n(P32(pool + POOL_LIST_N), n + 1, __ATOMIC_RELEASE);
    if (__atomic_load_n(&p->open, __ATOMIC_SEQ_CST)) bump(&p->seq), bump(&p->evt);
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
    __atomic_store_n(&p->open, 1, __ATOMIC_SEQ_CST);
    bump(&p->seq);
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
    for (;;) {
        uint32_t e = __atomic_load_n(&p->evt, __ATOMIC_ACQUIRE);
        walk(c, fns, p, w0, 0, &k, true);
        uint32_t n = __atomic_load_n(P32(pool + POOL_LIST_N), __ATOMIC_ACQUIRE), cap = LD32(pool + POOL_LIST_CAP);
        for (; j < n; j++) {
            uint64_t g = LD64(LD64(pool + POOL_LIST) + 8ull * (j < cap ? j : 0));
            if ((LD32(g + G_MASK) & valid) && __atomic_load_n(P32(g + G_DONE), __ATOMIC_ACQUIRE) < (uint32_t)call1(c, vfn(g, V_TOTAL), g))
                break;
        }
        if ((j == n && k == n) || g_rt_unwind) break;
        if (k == __atomic_load_n(P32(pool + POOL_LIST_N), __ATOMIC_ACQUIRE)) wait_change(&p->evt, e, true);
    }
    run_end(w0);
    __atomic_store_n(&p->open, 0, __ATOMIC_SEQ_CST);
    for (uint32_t b; (b = __atomic_load_n(&p->busy, __ATOMIC_SEQ_CST)) && !g_rt_unwind;) wait_change(&p->busy, b, true);
    uint32_t n = LD32(pool + POOL_LIST_N);
    if (g_check) check_round(c, pool, n);
    __atomic_store_n(P32(pool + POOL_LIST_N), 0, __ATOMIC_RELEASE);
    __atomic_add_fetch(&p->gen, 1, __ATOMIC_RELEASE);
    return 1;
}
