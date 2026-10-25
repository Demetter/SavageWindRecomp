// Exercise the same core arbitration used by kernel.c without game files.
#include <assert.h>
#include <stdatomic.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "../runtime/kernel.h"
volatile int g_rt_unwind;
static int test_b2;  // SAVAGEWIND_B2's bits as kernel.c would return them
#define CORE_B2_BITS() test_b2
#include "../runtime/kernel_core.inc"

static _Atomic int inside, finished;
static int order[128], norder;
static unsigned counter;
static int loops;

static void init_thread(KThread *t, int id, int prio) {
    memset(t, 0, sizeof *t);
    t->id = id; t->prio = prio; t->core_held = -1;
    pthread_cond_init(&t->cond, NULL);
    pthread_cond_init(&t->core_cond, NULL);
}

static void *run(void *arg) {
    KThread *t = arg;
    for (int i = 0; i < loops; i++) {
        core_acquire(t);
        if (g_rt_unwind) { core_release(t); break; }
        assert(atomic_fetch_add(&inside, 1) == 0);
        if (!i) order[norder++] = (int)t->id;
        counter++;  // deliberately not atomic: core exclusion must protect it
        assert(atomic_fetch_sub(&inside, 1) == 1);
        core_release(t);
    }
    atomic_fetch_add(&finished, 1);
    return NULL;
}

static void wait_count(int n, bool queued) {
    for (int i = 0; i < 5000; i++) {
        pthread_mutex_lock(&g_cores[0].m);
        int have = queued ? g_cores[0].nwaiting + g_cores[0].overflow : atomic_load(&finished);
        pthread_mutex_unlock(&g_cores[0].m);
        if (have == n) return;
        usleep(1000);
    }
    fprintf(stderr, "handoff timed out (%s %d)\n", queued ? "queued" : "finished", n);
    abort();
}

static void scenario(int n, bool change_priority, bool stop, int iterations) {
    KThread owner, t[80];
    pthread_t threads[80];
    init_thread(&owner, 100, 5);
    atomic_store(&finished, 0); norder = 0; counter = 0; loops = iterations;
    core_acquire(&owner);
    for (int i = 0; i < n; i++) {
        init_thread(&t[i], i, 20 + (n <= 8 ? i % 3 : 0));
        assert(!pthread_create(&threads[i], NULL, run, &t[i]));
        wait_count(i + 1, true);  // deterministic FIFO tickets
    }
    if (change_priority) {
        t[n - 1].prio = 1;
        if (core_direct_wake()) t[n - 1].core = 2;  // still waiting on core 0
        core_priority_changed(&t[n - 1]);
        assert(g_core_best_wait[0] == 1);
    }
    if (stop) {
        pthread_mutex_lock(&g_cores[0].m);
        g_rt_unwind = 1;
        core_wake_all(&g_cores[0]);
        pthread_mutex_unlock(&g_cores[0].m);
    }
    core_release(&owner);
    wait_count(n, false);
    for (int i = 0; i < n; i++) {
        pthread_join(threads[i], NULL);
        pthread_cond_destroy(&t[i].cond); pthread_cond_destroy(&t[i].core_cond);
    }
    if (!stop) {
        assert(counter == (unsigned)(n * iterations));
        if (iterations == 1 && n == 8 && !change_priority) {
            const int expected[] = {0,3,6,1,4,7,2,5};
            assert(!memcmp(order, expected, sizeof expected));
        }
        if (change_priority) assert(order[0] == n - 1);
    }
    assert(!g_cores[0].owner && !g_cores[0].nwaiting && !g_cores[0].overflow);
    g_rt_unwind = 0;
    pthread_cond_destroy(&owner.cond); pthread_cond_destroy(&owner.core_cond);
}

// A waiter's priority changes while the core is free and the waiter it was handed to has not taken it yet: the new
// best waiter must still get it (B2 bit 2 wakes only that one).
static void scenario_free_core_priority(void) {
    KThread owner, t[2];
    pthread_t threads[2];
    init_thread(&owner, 100, 5);
    atomic_store(&finished, 0); norder = 0; counter = 0; loops = 1;
    core_acquire(&owner);
    for (int i = 0; i < 2; i++) {
        init_thread(&t[i], i, 20 + i);
        assert(!pthread_create(&threads[i], NULL, run, &t[i]));
        wait_count(i + 1, true);
    }
    pthread_mutex_lock(&g_cores[0].m);  // let the core go without waking anyone, as if the woken one were still on its way
    g_cores[0].owner = NULL;
    owner.core_held = -1;
    pthread_mutex_unlock(&g_cores[0].m);
    t[1].prio = 1;
    core_priority_changed(&t[1]);
    wait_count(2, false);
    for (int i = 0; i < 2; i++) {
        pthread_join(threads[i], NULL);
        pthread_cond_destroy(&t[i].cond); pthread_cond_destroy(&t[i].core_cond);
    }
    assert(order[0] == 1 && counter == 2);
    assert(!g_cores[0].owner && !g_cores[0].nwaiting && !g_cores[0].overflow);
    pthread_cond_destroy(&owner.cond); pthread_cond_destroy(&owner.core_cond);
}

int main(void) {
    for (test_b2 = 0; test_b2 <= 2; test_b2 += 2) {
        scenario(8, false, false, 1);       // priority, then FIFO
        scenario(8, true, false, 1);        // a blocked waiter's priority changes
        scenario(8, false, false, 1000);    // exclusive execution under repeated handoff
        scenario(80, false, false, 1);     // overflow falls back without losing a waiter
        scenario(8, false, true, 1);        // all sleepers escape at reboot
        scenario_free_core_priority();      // a priority change on a free core still hands it over
    }
    puts("core scheduling: priority, FIFO, 8000 handoffs, overflow, shutdown, free-core priority change passed "
         "(B2 targeted priority wakes off and on)");
    return 0;
}
