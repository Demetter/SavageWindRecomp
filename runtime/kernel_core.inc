// Core arbitration, included by kernel.c and the standalone stress test.
typedef struct {
    pthread_mutex_t m;
    pthread_cond_t cv;
    KThread *owner;
    KThread *waiting[64];
    int nwaiting;
    uint64_t next_ticket;
    int overflow;  // more waiters than the tracked array: retain broadcast arbitration
} Core;
static Core g_cores[4] = {
    {PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER, NULL, {0}, 0, 0},
    {PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER, NULL, {0}, 0, 0},
    {PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER, NULL, {0}, 0, 0},
    {PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER, NULL, {0}, 0, 0},
};
static int g_cores_on = -1;
// SAVAGEWIND_B2's bits (kernel.c): 2 wakes only the best waiter of a free core on a priority change. The standalone
// test sets its own.
#ifndef CORE_B2_BITS
#define CORE_B2_BITS() 0
#endif
// The best (lowest) priority among the threads waiting for each core, INT_MAX with none: what dispatch.c checks at every
// indirect call (kernel_preempt). SAVAGEWIND_NO_PREEMPT=1: kept at INT_MAX (preemption at syscalls only, as before).
// One cache line per core, written only when it changes: every indirect call of every guest thread reads its core's
// entry, and one shared line rewritten at each core hand-over missed in their caches (Kakariko 3.85 -> 3.82 ms a frame).
CoreBestWait g_core_best_wait[4] = {{INT_MAX}, {INT_MAX}, {INT_MAX}, {INT_MAX}};
static void core_best_update(Core *k) {  // (k->m held)
    static int off = -1;
    if (off < 0) off = getenv("SAVAGEWIND_NO_PREEMPT") != NULL;
    int best = INT_MAX;
    for (int i = 0; i < k->nwaiting && !off; i++)
        if (k->waiting[i]->prio < best) best = k->waiting[i]->prio;
    if (__atomic_load_n(&g_core_best_wait[k - g_cores].v, __ATOMIC_RELAXED) != best)
        __atomic_store_n(&g_core_best_wait[k - g_cores].v, best, __ATOMIC_RELAXED);
}

static bool cores_on(void) {
    if (g_cores_on < 0) g_cores_on = !(getenv("SAVAGEWIND_CORES") && !strcmp(getenv("SAVAGEWIND_CORES"), "0"));
    return g_cores_on;
}

// Whether t is the waiter that gets the core next. Caller holds k->m.
static bool core_next(Core *k, KThread *t) {
    for (int i = 0; i < k->nwaiting; i++) {
        KThread *o = k->waiting[i];
        if (o != t && (o->prio < t->prio || (o->prio == t->prio && o->core_ticket < t->core_ticket))) return false;
    }
    return true;
}

// The waiter that gets the core next: the best priority, then the earliest ticket. Caller holds k->m.
static KThread *core_best_waiter(Core *k) {
    KThread *best = k->nwaiting ? k->waiting[0] : NULL;
    for (int i = 1; i < k->nwaiting; i++) {
        KThread *o = k->waiting[i];
        if (o->prio < best->prio || (o->prio == best->prio && o->core_ticket < best->core_ticket)) best = o;
    }
    return best;
}

// Priority changes and shutdown must reach all waiters. Ordinary releases wake
// just the best one; it still rechecks priority under the mutex before taking
// the token, so a newly runnable higher-priority thread can go first.
static bool core_direct_wake(void) {
    static int on = -1;
    int value = __atomic_load_n(&on, __ATOMIC_RELAXED);
    if (value < 0) {
        const char *e = getenv("SAVAGEWIND_CORE_DIRECT_WAKE");
        value = !e || atoi(e) != 0;  // set to 0 to compare with the original broadcast wakeups
        __atomic_store_n(&on, value, __ATOMIC_RELAXED);
    }
    return value != 0;
}

static void core_wake_all(Core *k) {
    pthread_cond_broadcast(&k->cv);
    if (core_direct_wake())
        for (int i = 0; i < k->nwaiting; i++) pthread_cond_signal(&k->waiting[i]->core_cond);
}

static void core_acquire(KThread *t) {
    if (!t || !cores_on() || t->core_held >= 0 || t->off_cores) return;
    static int one_core = -1;  // (debug) SAVAGEWIND_ONE_CORE=1: every thread on core 0, one guest thread at a time
    if (one_core < 0) one_core = getenv("SAVAGEWIND_ONE_CORE") != NULL;
    int idx = one_core ? 0 : t->core & 3;
    Core *k = &g_cores[idx];
    pthread_mutex_lock(&k->m);
    t->core_ticket = k->next_ticket++;
    bool tracked = k->nwaiting < 64;
    if (tracked) k->waiting[k->nwaiting++] = t;
    else k->overflow++;
    core_best_update(k);
    pthread_cond_t *cv = core_direct_wake() && tracked ? &t->core_cond : &k->cv;
    while ((k->owner || !core_next(k, t)) && !g_rt_unwind) pthread_cond_wait(cv, &k->m);
    if (!tracked) k->overflow--;
    for (int i = 0; i < k->nwaiting; i++)
        if (k->waiting[i] == t) { k->waiting[i] = k->waiting[--k->nwaiting]; break; }
    core_best_update(k);
    if (!k->owner) k->owner = t;  // (during a reboot the core may stay with another thread: run anyway)
    t->core_held = k->owner == t ? idx : -1;
    pthread_mutex_unlock(&k->m);
}

static void core_release(KThread *t) {
    if (!t || t->core_held < 0) return;
    Core *k = &g_cores[t->core_held];
    pthread_mutex_lock(&k->m);
    if (k->owner == t) k->owner = NULL;
    t->core_held = -1;
    if (!core_direct_wake() || k->overflow || g_rt_unwind) core_wake_all(k);
    else if (k->nwaiting) pthread_cond_signal(&core_best_waiter(k)->core_cond);
    pthread_mutex_unlock(&k->m);
}

static void core_priority_changed(KThread *t) {
    // Another thread may change a waiter's affinity before it acquires its old
    // core. Direct wakeups must also reach that old core (and ONE_CORE's core 0),
    // otherwise a newly best waiter can sleep while the core is unowned.
    // (B2 bit 2) Only a core nobody holds needs a wakeup, and only for its best waiter: a held core goes to the best
    // waiter when it is let go, and a waiter better than the holder takes it at the holder's next preemption point
    // (g_core_best_wait). Waking every waiter of all four cores on each mutex hand-over cost OS wakeups for nothing.
    int first = core_direct_wake() ? 0 : t->core & 3;
    int last = core_direct_wake() ? 4 : first + 1;
    bool targeted = (CORE_B2_BITS() & 2) && core_direct_wake();
    for (int i = first; i < last; i++) {
        Core *k = &g_cores[i];
        pthread_mutex_lock(&k->m);
        core_best_update(k);
        if (!targeted || k->overflow) core_wake_all(k);
        else if (!k->owner && k->nwaiting) pthread_cond_signal(&core_best_waiter(k)->core_cond);
        pthread_mutex_unlock(&k->m);
    }
}

