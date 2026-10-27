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

// The slot idx of obj's array (rt_percore_block), checked: a core the block has no slot for is a bug (on the Switch it
// would write past the game's array into the next field).
uint64_t rt_percore_slot(uint64_t obj, uint32_t field, uint32_t stride, uint32_t switch_slots, int with_system,
                         int64_t idx) {
    uint64_t block = rt_percore_block(obj, field, stride, switch_slots, with_system);
    int64_t slots = rt_game_cores() + (with_system ? 1 : 0);
    if (idx < 0 || idx >= slots)
        FATAL("percore: %llx +%#x: core %lld, but %lld slots\n", (unsigned long long)obj, field, (long long)idx,
              (long long)slots);
    return block + (uint64_t)idx * stride;
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

// ---- #10: the command memory arenas (one object, at +0x90 of its owner). Each game core records its command buffers
// into its own memory: a 0x28-byte record per core at +0x88 (+0 block count, +8 blocks of 0x18 bytes {GPU address,
// u32, CPU address} - one per frame in flight, +0x10 / +0x14 command memory used / size of a block, +0x18 control
// memory base, +0x20 / +0x24 control used / size). The game's setup, per-frame reset and destructor are unrolled for
// its three records; these run them over every core's. (The constructor's zeroing of the game's own records needs no
// counterpart: the block is made zeroed, by the setup, after the constructor ran.)
extern void guest_call(Cpu *c, uint64_t addr);
#define ARENA_RECS(obj) rt_percore_block(obj, 0x88, 0x28, 3, 0)

static uint64_t sbfx34(uint64_t v) { return (uint64_t)((int64_t)(v << 30) >> 30); }

// f_7000ef3098 (1.9.0): setup(obj x0, command bytes per block x1, control bytes per core x2, command buffers w3,
// blocks per core w4, heap x5). fn: the game's [0] malloc(size, heap, align), [1] memory pool init(pool, size, heap,
// 4, 1), [2] pool address(out, pool, offset) -> {GPU address, u32, CPU address}, [3] the variable holding the
// graphics API's "CPU address of a memory pool" function, [4] pool offset(pool), [5] a record's block list(record,
// count, heap), [6] the command buffers(array, count, heap, 8). Both memories are sized for every core; the blocks
// are carved out of the pool core after core, one running offset per pool (the second pool, +0x40, takes every other
// block when it has memory).
void rt_arena_setup(Cpu *c, const uint64_t *fn) {
    uint64_t obj = c->x[0], ctl = c->x[2], heap = c->x[5], blk = (c->x[1] + 3) & ~3ull;
    uint32_t nbuf = (uint32_t)c->x[3];
    uint8_t per = (uint8_t)c->x[4];
    int n = rt_game_cores();
    ST8(obj + 0x110, per);
    uint64_t total = blk * per * (uint64_t)n;
    if (ctl) {
        c->x[0] = ctl * (uint64_t)n, c->x[1] = heap, c->x[2] = 8;
        guest_call(c, fn[0]);
        ST64(obj + 0x78, c->x[0]);
    }
    if (total) {
        uint64_t sp = c->sp, out = (c->sp -= 0x40);  // (the pool address's 0x18-byte result)
        c->x[0] = obj + 8, c->x[1] = sbfx34(total), c->x[2] = heap, c->x[3] = 4, c->x[4] = 1;
        guest_call(c, fn[1]);
        c->x[0] = out, c->x[1] = obj + 8, c->x[2] = 0;
        guest_call(c, fn[2]);
        c->x[0] = LD64(obj + 0x20);
        guest_call(c, LD64(fn[3]));
        uint64_t cpu = c->x[0];
        c->x[0] = obj + 8;
        guest_call(c, fn[4]);
        uint64_t mem = cpu + c->x[0];
        uint32_t words = (uint32_t)(LD64(obj + 0x18) >> 2);  // (the pool's size: filled with the game's pattern)
        if ((int32_t)words >= 1)
            for (uint32_t i = 0; i < words; i++) ST32(mem + 4ull * i, 0xbeef2929u);
        uint64_t recs = ARENA_RECS(obj), pos[2] = {0, 0};
        int pool = 0;
        for (int i = 0; i < n; i++) {
            uint64_t r = recs + 0x28ull * (uint64_t)i;
            ST32(r + 0x10, 0), ST32(r + 0x14, blk);
            c->x[0] = r, c->x[1] = LD8(obj + 0x110), c->x[2] = heap;
            guest_call(c, fn[5]);
            ST64(r + 0x18, LD64(obj + 0x78) + ctl * (uint64_t)i), ST32(r + 0x20, 0), ST32(r + 0x24, ctl);
            uint32_t count = LD32(r);
            uint64_t blocks = LD64(r + 8);
            for (uint32_t b = 0; b < count; b++) {
                uint64_t o = pos[pool];
                c->x[0] = out, c->x[1] = pool ? obj + 0x40 : obj + 8, c->x[2] = sbfx34(o) & ~3ull;
                guest_call(c, fn[2]);
                uint64_t e = blocks + 0x18ull * b;
                ST64(e, LD64(out)), ST32(e + 8, LD32(out + 8)), ST64(e + 0x10, LD64(out + 0x10));
                pos[pool] = o + blk;
                pool = (int32_t)(uint32_t)(LD64(obj + 0x50) >> 2) > 0 ? !pool : 0;
            }
        }
        c->sp = sp;
    }
    if ((int32_t)nbuf >= 1) {
        c->x[0] = obj + 0x100, c->x[1] = nbuf, c->x[2] = heap, c->x[3] = 8;
        guest_call(c, fn[6]);
    }
    LOG("[percore] command arenas: %d cores x %u blocks of %#llx command bytes, %#llx control bytes each; %u command "
        "buffers\n", n, per, (unsigned long long)blk, (unsigned long long)ctl, nbuf);
}

// f_7000ef33e8 (1.9.0): a frame's end (w1 bit 0: on to the next block) - the peak use over every core, each core's use
// back to 0, the command buffer counter to 0.
void rt_arena_reset(Cpu *c) {
    uint64_t obj = c->x[0], recs = ARENA_RECS(obj);
    int n = rt_game_cores();
    for (int i = 0; i < n; i++) {
        uint64_t r = recs + 0x28ull * (uint64_t)i;
        uint32_t cmd = LD32(r + 0x10), ctl = LD32(r + 0x20);
        if (LD64(obj + 0x118) < cmd) ST64(obj + 0x118, cmd);
        if (LD64(obj + 0x120) < ctl) ST64(obj + 0x120, ctl);
        ST32(r + 0x10, 0), ST32(r + 0x20, 0);
    }
    ST32(obj + 0x80, 0);
    if (c->x[1] & 1) {
        uint32_t next = LD8(obj + 0x111) + 1u, per = LD8(obj + 0x110);
        ST8(obj + 0x111, per ? next % per : next);  // (the game's udiv by 0 gives 0: next as it is)
    }
}

// The destructor f_7000ef2f54 (1.9.0): every core's block list (made with an 8-byte header) back to the heap (free_fn).
void rt_arena_free_blocks(Cpu *c, uint64_t obj, uint64_t free_fn) {
    uint64_t recs = ARENA_RECS(obj);
    int n = rt_game_cores();
    for (int i = 0; i < n; i++) {
        uint64_t r = recs + 0x28ull * (uint64_t)i, blocks = LD64(r + 8);
        if (!blocks) continue;
        c->x[0] = blocks - 8;
        guest_call(c, free_fn);
        ST64(r + 8, 0), ST32(r, 0);
    }
}
