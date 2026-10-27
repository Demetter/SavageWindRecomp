// Per-core arrays with a slot for every core (BOTW-JOBS-NATIVE.md). The game keeps some per-core state as arrays inline
// in an object, sized for the Switch: sead's 3 game cores, or all 4 cores (then the last one is the system core's
// threads'). The generated code's accesses to such an array (tools/patches.py) take its base from here instead: one
// block per object and array, holding a slot for each of the game's cores and, where the Switch's array had one, a
// last slot for the system core's threads - so the system slot's index is the game's core count. Blocks live as long as
// the game (the objects they serve are singletons made at boot).
//
// (check) SAVAGEWIND_PERCORE_POISON=1: when a block is made, the object's own array is filled with a pattern that
// faults on use (counts of 0x100, pointers into the null page), and every ~600 frames the pattern is checked: a missed
// read faults at once, a missed write is logged with the array and offset.
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "../hle.h"
#include "../cpu.h"

extern int rt_game_cores(void);

#define PERCORE_MAX 64
#define POISON 0x0000000000000100ull
static struct {
    uint64_t obj, block;
    uint32_t field, bytes;  // (the object's own array: obj + field, bytes long)
} g_blocks[PERCORE_MAX];
static int g_nblocks;  // (published with release after the entry is filled; entries never change afterwards)
static pthread_mutex_t g_m = PTHREAD_MUTEX_INITIALIZER;
static int g_poison = -1;

// The block for obj's array at +field of slots of `stride` bytes, which the Switch sized for `switch_slots` cores
// (with_system: its last slot is the system core's threads').
uint64_t rt_percore_block(uint64_t obj, uint32_t field, uint32_t stride, uint32_t switch_slots, int with_system) {
    int n = __atomic_load_n(&g_nblocks, __ATOMIC_ACQUIRE);
    for (int i = 0; i < n; i++)
        if (g_blocks[i].obj == obj && g_blocks[i].field == field) return g_blocks[i].block;
    pthread_mutex_lock(&g_m);
    n = g_nblocks;
    for (int i = 0; i < n; i++)
        if (g_blocks[i].obj == obj && g_blocks[i].field == field) {
            pthread_mutex_unlock(&g_m);
            return g_blocks[i].block;
        }
    if (n == PERCORE_MAX) FATAL("percore: more than %d per-core arrays\n", PERCORE_MAX);
    size_t slots = (size_t)rt_game_cores() + (with_system ? 1 : 0);
    void *b = calloc(1, slots * stride + 64);
    if (!b) FATAL("percore: out of memory\n");
    uint64_t block = ((uint64_t)(uintptr_t)b + 63) & ~63ull;
    if (g_poison < 0) g_poison = getenv("SAVAGEWIND_PERCORE_POISON") && atoi(getenv("SAVAGEWIND_PERCORE_POISON"));
    uint32_t bytes = switch_slots * stride;
    if (g_poison)
        for (uint32_t o = 0; o + 8 <= bytes; o += 8) *(uint64_t *)(uintptr_t)(obj + field + o) = POISON;
    g_blocks[n].obj = obj, g_blocks[n].block = block, g_blocks[n].field = field, g_blocks[n].bytes = bytes;
    __atomic_store_n(&g_nblocks, n + 1, __ATOMIC_RELEASE);
    pthread_mutex_unlock(&g_m);
    LOG("[percore] %llx +%#x: %zu slots of %#x (the game's %u%s)%s\n", (unsigned long long)obj, field, slots, stride,
        switch_slots, with_system ? ", the last the system core's" : "", g_poison ? "; its own array poisoned" : "");
    return block;
}

// (check) once a frame (runtime/hle/framepace.c): every ~600 frames, the poisoned arrays still poisoned?
void rt_percore_check(void) {
    static uint64_t frames;
    if (g_poison <= 0 || ++frames % 600) return;
    int n = __atomic_load_n(&g_nblocks, __ATOMIC_ACQUIRE);
    static uint8_t reported[PERCORE_MAX];
    for (int i = 0; i < n; i++) {
        if (reported[i]) continue;
        for (uint32_t o = 0; o + 8 <= g_blocks[i].bytes; o += 8)
            if (*(uint64_t *)(uintptr_t)(g_blocks[i].obj + g_blocks[i].field + o) != POISON) {
                LOG("[percore] MISSED WRITE: %llx +%#x, the game's own array written at +%#x\n",
                    (unsigned long long)g_blocks[i].obj, g_blocks[i].field, o);
                reported[i] = 1;
                break;
            }
    }
}

// ---- #8: the command contexts (0x288 bytes; the game's four at +0x30: three game cores and the system core's)
// f_7000ed626c (1.9.0) as a loop over every context: +0xaf0 = v, and in each context the +0x88 of its entry 1 (when it
// has more than two) or entry 0 (0xa0-byte entries: +0 count, +8 array) = v.
void rt_ctx_set_entries(Cpu *c) {
    uint64_t self = c->x[0], v = c->x[1];
    *(uint64_t *)(uintptr_t)(self + 0xaf0) = v;
    uint64_t blk = rt_percore_block(self, 0x30, 0x288, 4, 1);
    int slots = rt_game_cores() + 1;
    for (int i = 0; i < slots; i++) {
        uint64_t ctx = blk + 0x288ull * i;
        uint32_t count = *(uint32_t *)(uintptr_t)ctx;
        uint64_t arr = *(uint64_t *)(uintptr_t)(ctx + 8);
        *(uint64_t *)(uintptr_t)(arr + (count > 2 ? 0x140 : 0) + 0x88) = v;
    }
}
