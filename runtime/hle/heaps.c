// Game heaps (engine limits). The game carves its memory into SDK heaps of fixed sizes: nn::mem::StandardAllocator
// (general), nn::lmem unit heaps (fixed-size blocks: object pools) and frame heaps. A heap that runs out returns
// NULL and the game crashes further on (bushsanity: ~450 actors around Link -> null weak_ptr in f_70010d0f00).
// These wrappers call the SDK's own implementation (loader_find_symbol), log heap creation (SAVAGEWIND_LOG_HEAPS=1)
// and every allocation that fails (once per caller), and can grow heaps (heap_scale) - a recomp has memory to spare.
#include <pthread.h>
#ifdef _WIN32
#include <malloc.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../hle.h"

uint64_t loader_find_symbol(const char *name);
void guest_call(Cpu *c, uint64_t addr);

static uint64_t orig(const char *name, uint64_t *cache) {
    if (!*cache) *cache = loader_find_symbol(name);
    if (!*cache) FATAL("heaps: no SDK implementation of %s\n", name);
    return *cache;
}

static void fail_once(const char *what, uint64_t lr, uint64_t size) {
    static uint64_t seen[64];
    static int n;
    for (int i = 0; i < n; i++) if (seen[i] == lr) return;
    if (n < 64) seen[n++] = lr;
    LOG("[heaps] %s FAILED (size %#llx) from lr %llx\n", what, (unsigned long long)size, (unsigned long long)lr);
}

#define CALL(sym)                                    \
    do {                                             \
        static uint64_t a_;                          \
        uint64_t lr_ = c->x[30];                     \
        guest_call(c, orig(sym, &a_));               \
        c->x[30] = lr_;                              \
    } while (0)

// Unit heaps are fixed pools (e.g. 256 x 0x50-byte objects per loaded area: bushsanity's extra actors ran out at
// ~450 actors around Link). Every unit heap gets UNIT_SCALE times the units in a buffer of ours (the game's own
// buffer stays unused); freed again in DestroyUnitHeap. SAVAGEWIND_UNIT_HEAP_SCALE overrides (1 = the game's sizes).
typedef struct { uint64_t head; void *mem; } UnitBuf;
static UnitBuf g_unit[256];
static pthread_mutex_t g_unit_lock = PTHREAD_MUTEX_INITIALIZER;

// Reboot: our unit-heap buffers (host memory, not in the guest window) go with the game that made them.
RT_ON_RESET(heaps_reset) {
    for (int i = 0; i < 256; i++)
        if (g_unit[i].mem) {
#ifdef _WIN32
            _aligned_free(g_unit[i].mem);
#else
            free(g_unit[i].mem);
#endif
            g_unit[i].mem = NULL;
        }
}

// The game's memory heap (tools/patches.py _init_heap_size): BOTW's nninitStartup asks for a fixed 0xC3000000, the
// Switch's application memory (0xCD500000) less 0xA500000 for code, stacks and the system. Here the application memory
// is the PC's (runtime/rt.h HEAP_MAX, GetInfo 6 / 21) and the game keeps the same 0xA500000 back.
uint64_t rt_memory_heap_size(void) {
    const char *e = getenv("SAVAGEWIND_MEMORY_HEAP");  // ((comparison) a size in MB, e.g. 3120 for the Switch's)
    return e && atoi(e) > 0 ? (uint64_t)atoi(e) << 20 : HEAP_MAX - 0x2B00000ull - 0xA500000ull;
}
// sead's root heap (tools/patches.py _root_heap_size): the game asks for a fixed 0xC1000000 - its memory heap less the
// 32 MB block nninitStartup gives malloc. Here the same: the PC's memory heap less that block. Everything below the root
// ("KingSystem", the stage heaps, "the rest" heaps) is carved from it by the game.
uint64_t rt_root_heap_size(void) { return rt_memory_heap_size() - 0x2000000ull; }

// (check) sead's heaps, as made: the base constructor (tools/patches.py _heap_ctor) records each one - this, name,
// parent, start, size (sead::Heap::Heap's arguments) - and rt_heap_dump writes every one still alive with its free
// size and largest allocation, from the game's own functions (vtable + 0x78 getFreeSize, + 0x80 getMaxAllocatableSize).
// The kernel calls it where a thread's stack could not be allocated (runtime/kernel.c MapMemory), and where
// SAVAGEWIND_HEAP_DUMP_AT=<guest return address> is in a mapping's call chain (the same moment in a run that works).
typedef struct { uint64_t heap, name, parent, size; } HeapRec;
static HeapRec *g_heaps;
static size_t g_nheaps, g_capheaps;
static pthread_mutex_t g_heaps_m = PTHREAD_MUTEX_INITIALIZER;

void rt_heap_made(Cpu *c) {
    pthread_mutex_lock(&g_heaps_m);
    if (g_nheaps == g_capheaps) {
        size_t cap = g_capheaps ? g_capheaps * 2 : 1024;
        HeapRec *n = realloc(g_heaps, cap * sizeof *n);
        if (!n) { pthread_mutex_unlock(&g_heaps_m); return; }
        g_heaps = n, g_capheaps = cap;
    }
    uint64_t name = c->x[1] ? LD64(c->x[1] + 8) : 0;  // (sead::SafeString: the characters' pointer at +8)
    g_heaps[g_nheaps++] = (HeapRec){c->x[0], name, c->x[2], c->x[4]};
    pthread_mutex_unlock(&g_heaps_m);
}

static void heap_name(uint64_t s, char *out, size_t size) {
    size_t i = 0;
    if (s >= GUEST_LO && s < GUEST_HI)
        for (; i + 1 < size; i++) {
            char ch = (char)LD8(s + i);
            if (ch < 0x20 || ch > 0x7e) break;
            out[i] = ch;
        }
    out[i] = 0;
}

// One heap and, below it, its children (sead::Heap: parent +0x8, name +0x28, start +0x38, size +0x40, children: a list
// head at +0x50, count +0x60, node offset +0x64 = 0x68), each line indented by depth.
static void dump_heap(Cpu *c, uint64_t h, int depth, int *n) {
    if (++*n > 8000 || depth > 32) return;
    uint64_t vt = LD64(h);  // (a heap's class: its vtable and size functions in the game's modules)
    if (vt < MAIN_BASE || vt >= MODULES_END || LD64(vt + 0x78) < MAIN_BASE || LD64(vt + 0x78) >= MODULES_END ||
        LD64(vt + 0x80) < MAIN_BASE || LD64(vt + 0x80) >= MODULES_END) {
        LOG("[heap] %*s%llx: not a heap (vtable %llx)\n", 2 * depth, "", (unsigned long long)h, (unsigned long long)vt);
        return;
    }
    c->x[0] = h;
    guest_call(c, LD64(vt + 0x78));
    uint64_t free = c->x[0];
    c->x[0] = h, c->x[1] = 8;
    guest_call(c, LD64(vt + 0x80));
    uint64_t max = c->x[0];
    char name[64];
    heap_name(LD64(h + 0x28), name, sizeof name);
    LOG("[heap] %*s%llx \"%s\": start %llx size %#llx free %#llx largest %#llx children %u\n", 2 * depth, "",
        (unsigned long long)h, name, (unsigned long long)LD64(h + 0x38), (unsigned long long)LD64(h + 0x40),
        (unsigned long long)free, (unsigned long long)max, LD32(h + 0x60));
    if (LD32(h + 0x64) != 0x68) return;
    uint64_t head = h + 0x50;
    for (uint64_t node = LD64(head + 8), k = 0; node != head && k < 8192; node = LD64(node + 8), k++) {
        if (node < GUEST_LO + 0x68 || node >= GUEST_HI || (node & 7)) break;  // (a list in flux: stop)
        dump_heap(c, node - 0x68, depth + 1, n);
    }
}

void rt_heap_dump(Cpu *c, const char *why) {
    pthread_mutex_lock(&g_heaps_m);
    uint64_t saved[31], sp = c->sp;  // (the caller - a syscall - keeps every register)
    memcpy(saved, c->x, sizeof saved);
    uint64_t roots[16];
    int nroots = 0, n = 0;
    for (size_t i = 0; i < g_nheaps; i++) {  // (the roots: up from every heap the base constructor made, still alive)
        HeapRec *r = &g_heaps[i];
        bool later = false;  // (an address made again later: only its latest heap is alive)
        for (size_t j = i + 1; j < g_nheaps && !later; j++) later = g_heaps[j].heap == r->heap;
        if (later || LD64(r->heap + 0x28) != r->name || LD64(r->heap + 8) != r->parent) continue;
        uint64_t p = r->heap;
        for (int k = 0; k < 64; k++) {
            uint64_t up = LD64(p + 8);
            if (!up || up < GUEST_LO || up >= GUEST_HI || (up & 7)) break;
            p = up;
        }
        bool known = false;
        for (int j = 0; j < nroots; j++) known |= roots[j] == p;
        if (!known && nroots < 16) roots[nroots++] = p;
    }
    LOG("[heap] dump (%s): %d root heaps\n", why, nroots);
    for (int j = 0; j < nroots; j++) dump_heap(c, roots[j], 0, &n);
    memcpy(c->x, saved, sizeof saved);
    c->sp = sp;
    pthread_mutex_unlock(&g_heaps_m);
}

// Every sead heap made with its own lock (tools/patches.py _heap_ctor): the game's threads run in parallel here, not one
// at a time per Switch core. SAVAGEWIND_HEAP_LOCKS=0 keeps the game's choice (for comparison).
int rt_heaps_locked(void) {
    static int on = -1;
    if (on < 0) { const char *e = ENV("SAVAGEWIND_HEAP_LOCKS"); on = !e || atoi(e) != 0; }
    return on;
}

static int unit_scale(void) {
    static int s = -1;
    if (s < 0) { const char *e = ENV("SAVAGEWIND_UNIT_HEAP_SCALE"); s = e ? atoi(e) : 4; if (s < 1) s = 1; }
    return s;
}

// nn::lmem::CreateUnitHeap(void *addr, size_t size, size_t unit, int align, int opt, HeapHead *head)
HLE_FUNC("_ZN2nn4lmem14CreateUnitHeapEPvmmiiPNS0_6detail8HeapHeadE", heap_create_unit) {
    uint64_t lr = c->x[30], size = c->x[1], unit = c->x[2];
    int k = unit_scale();
    void *mem = NULL;
    if (k > 1 && unit) {
        uint64_t big = size * (uint64_t)k;
#ifdef _WIN32
        mem = _aligned_malloc(big, 0x1000);
        if (mem) {
#else
        if (posix_memalign(&mem, 0x1000, big) == 0) {
#endif
            memset(mem, 0, big);
            c->x[0] = (uint64_t)(uintptr_t)mem;
            c->x[1] = big;
        } else {
            mem = NULL;
        }
    }
    uint64_t head = c->x[5];
    CALL("_ZN2nn4lmem14CreateUnitHeapEPvmmiiPNS0_6detail8HeapHeadE");
    if (mem) {
        pthread_mutex_lock(&g_unit_lock);
        for (int i = 0; i < 256; i++)
            if (!g_unit[i].mem) { g_unit[i].head = c->x[0] ? c->x[0] : head; g_unit[i].mem = mem; break; }
        pthread_mutex_unlock(&g_unit_lock);
    }
    if (ENV("SAVAGEWIND_LOG_HEAPS"))
        LOG("[heaps] unit heap %#llx bytes, unit %#llx (%llu units, x%d) from lr %llx -> %llx\n", (unsigned long long)size,
            (unsigned long long)unit, unit ? (unsigned long long)(size / unit) : 0ull, mem ? k : 1, (unsigned long long)lr,
            (unsigned long long)c->x[0]);
}

HLE_FUNC("_ZN2nn4lmem15DestroyUnitHeapEPNS0_6detail8HeapHeadE", heap_destroy_unit) {
    uint64_t head = c->x[0];
    CALL("_ZN2nn4lmem15DestroyUnitHeapEPNS0_6detail8HeapHeadE");
    pthread_mutex_lock(&g_unit_lock);
    for (int i = 0; i < 256; i++)
        if (g_unit[i].mem && g_unit[i].head == head) {
#ifdef _WIN32
            _aligned_free(g_unit[i].mem);
#else
            free(g_unit[i].mem);
#endif
            g_unit[i].mem = NULL;
            break;
        }
    pthread_mutex_unlock(&g_unit_lock);
}

HLE_FUNC("_ZN2nn4lmem20AllocateFromUnitHeapEPNS0_6detail8HeapHeadE", heap_alloc_unit) {
    uint64_t lr = c->x[30], head = c->x[0];
    CALL("_ZN2nn4lmem20AllocateFromUnitHeapEPNS0_6detail8HeapHeadE");
    if (!c->x[0]) fail_once("unit heap allocation", lr, head);
}

// nn::lmem::CreateFrameHeap(void *addr, size_t size, int opt)
HLE_FUNC("_ZN2nn4lmem15CreateFrameHeapEPvmi", heap_create_frame) {
    uint64_t lr = c->x[30], size = c->x[1];
    CALL("_ZN2nn4lmem15CreateFrameHeapEPvmi");
    if (ENV("SAVAGEWIND_LOG_HEAPS"))
        LOG("[heaps] frame heap %#llx bytes from lr %llx\n", (unsigned long long)size, (unsigned long long)lr);
}

HLE_FUNC("_ZN2nn4lmem21AllocateFromFrameHeapEPNS0_6detail8HeapHeadEmi", heap_alloc_frame) {
    uint64_t lr = c->x[30], size = c->x[1];
    CALL("_ZN2nn4lmem21AllocateFromFrameHeapEPNS0_6detail8HeapHeadEmi");
    if (!c->x[0]) fail_once("frame heap allocation", lr, size);
}

// nn::mem::StandardAllocator::Initialize(void *addr, size_t size[, bool])
HLE_FUNC("_ZN2nn3mem17StandardAllocator10InitializeEPvm", heap_std_init) {
    uint64_t lr = c->x[30], size = c->x[2];
    CALL("_ZN2nn3mem17StandardAllocator10InitializeEPvm");
    if (ENV("SAVAGEWIND_LOG_HEAPS"))
        LOG("[heaps] standard allocator %#llx bytes from lr %llx\n", (unsigned long long)size, (unsigned long long)lr);
}

HLE_FUNC("_ZN2nn3mem17StandardAllocator10InitializeEPvmb", heap_std_init_b) {
    uint64_t lr = c->x[30], size = c->x[2];
    CALL("_ZN2nn3mem17StandardAllocator10InitializeEPvmb");
    if (ENV("SAVAGEWIND_LOG_HEAPS"))
        LOG("[heaps] standard allocator %#llx bytes (b) from lr %llx\n", (unsigned long long)size, (unsigned long long)lr);
}

HLE_FUNC("_ZN2nn3mem17StandardAllocator8AllocateEm", heap_std_alloc) {
    uint64_t lr = c->x[30], size = c->x[1];
    CALL("_ZN2nn3mem17StandardAllocator8AllocateEm");
    if (!c->x[0] && size) fail_once("standard allocation", lr, size);
}

HLE_FUNC("_ZN2nn3mem17StandardAllocator8AllocateEmm", heap_std_alloc_a) {
    uint64_t lr = c->x[30], size = c->x[1];
    CALL("_ZN2nn3mem17StandardAllocator8AllocateEmm");
    if (!c->x[0] && size) fail_once("standard allocation (aligned)", lr, size);
}
