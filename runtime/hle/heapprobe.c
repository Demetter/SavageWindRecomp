// (research) Which threads use each sead heap whose lock is off (BOTW-JOBS-NATIVE.md step 3). On the Switch two threads
// pinned to one core never run at once, so the game leaves some heaps unlocked although several threads of one core
// use them; here, with threads no longer sharing emulated cores, those heaps need their own lock. A game library built
// with SAVAGEWIND_GEN_HEAP_PROBE=1 (tools/gen.py, tools/patches.py heap_probe_table) calls rt_heap_probe at every heap
// operation's lock-flag check (sead::Heap flags at +0xd8, bit 0) when the lock is off. Each heap a second thread
// reaches is logged, once per thread: the heap, its name, the threads that used it so far with their cores, and the
// check's address.
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "../cpu.h"
#include "../hle.h"
#include "../kernel.h"
#include "../rt.h"

uint32_t kernel_current_core(void);

typedef struct {
    uint64_t heap;
    uint32_t thread[8], core[8];
    int n;
} Rec;
static Rec g_tab[8192];  // (open addressing by heap address)
static pthread_mutex_t g_m = PTHREAD_MUTEX_INITIALIZER;

RT_ON_RESET(heapprobe_reboot_reset) { memset(g_tab, 0, sizeof g_tab); }

static void heap_name(uint64_t heap, char *out, size_t size) {
    uint64_t s = LD64(heap + 0x28);  // (sead::Heap's name: a SafeString at +0x20, its characters' pointer at +0x28)
    size_t i = 0;
    if (s >= 0x7000000000ull && s < 0x8000000000ull)
        for (; i + 1 < size; i++) {
            char ch = (char)LD8(s + i);
            if (ch < 0x20 || ch > 0x7e) break;
            out[i] = ch;
        }
    out[i] = 0;
}

void rt_heap_probe(Cpu *c, uint64_t heap, uint64_t pc) {
    (void)c;
    uint32_t th = kernel_current_thread_handle(), core = kernel_current_core();
    pthread_mutex_lock(&g_m);
    uint32_t i = (uint32_t)((heap >> 3) * 0x9e3779b97f4a7c15ull >> 51) & 8191;
    for (int probes = 0; probes < 8192 && g_tab[i].heap && g_tab[i].heap != heap; probes++) i = (i + 1) & 8191;
    Rec *r = &g_tab[i];
    if (!r->heap) r->heap = heap;
    bool known = false;
    for (int k = 0; k < r->n; k++) known |= r->thread[k] == th;
    if (r->heap == heap && !known && r->n < 8) {
        r->thread[r->n] = th, r->core[r->n] = core, r->n++;
        if (r->n >= 2) {
            char name[48], line[256];
            heap_name(heap, name, sizeof name);
            int len = snprintf(line, sizeof line, "[heap-probe] heap %llx \"%s\" unlocked, used by", (unsigned long long)heap,
                               name);
            for (int k = 0; k < r->n && len < (int)sizeof line - 24; k++)
                len += snprintf(line + len, sizeof line - len, " %x/c%u", r->thread[k], r->core[k]);
            LOG("%s (at %llx)\n", line, (unsigned long long)pc);
        }
    }
    pthread_mutex_unlock(&g_m);
}
