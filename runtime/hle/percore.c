// Per-core arrays with a slot for every core (BOTW-JOBS-NATIVE.md). The game keeps some per-core state as arrays inline
// in an object, sized for the Switch: sead's 3 game cores, or all 4 cores (then the last one is the system core's
// threads'). The generated code's accesses to such an array (tools/patches.py) take its base from here instead: one
// block per object and array, holding a slot for each of the game's cores and, where the Switch's array had one, a
// last slot for the system core's threads - so the system slot's index is the game's core count. A block is found by
// its object and field in a hash table (readers take no lock: an entry is published by its object, with release, and a
// table a bigger one replaces stays readable). An object made again at the same address gets its block back (the
// patched constructors zero it); blocks are not freed.
//
// (check) SAVAGEWIND_PERCORE_POISON=1: when a block is made, the object's own array is filled with a pattern that
// faults on use (counts of 0x100, pointers into the null page; byte arrays 0xa5), and every ~600 frames the pattern is
// checked: a missed read of a count or pointer faults at once, a missed write is logged with the array and offset.
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "../hle.h"
#include "../cpu.h"

extern int rt_game_cores(void);

#define POISON 0x0000000000000100ull
typedef struct {
    uint64_t obj;  // (0: free; written last, with release)
    uint64_t block;
    uint32_t field, bytes;  // (the object's own array: obj + field, bytes long)
    uint8_t reported;       // (check: a missed write logged)
} Entry;
typedef struct {
    uint32_t mask, used;
    Entry e[];
} Table;
static Table *g_tab;  // (published with release)
static pthread_mutex_t g_m = PTHREAD_MUTEX_INITIALIZER;
static int g_poison = -1;

static uint32_t slot_of(const Table *t, uint64_t obj, uint32_t field) {
    return (uint32_t)(((obj ^ ((uint64_t)field << 40)) * 0x9e3779b97f4a7c15ull) >> 32) & t->mask;
}

static Entry *find(Table *t, uint64_t obj, uint32_t field) {
    if (!t) return NULL;
    for (uint32_t i = slot_of(t, obj, field);; i = (i + 1) & t->mask) {
        uint64_t o = __atomic_load_n(&t->e[i].obj, __ATOMIC_ACQUIRE);
        if (!o) return NULL;
        if (o == obj && t->e[i].field == field) return &t->e[i];
    }
}

// (with g_m) a free entry for (obj, field), the table grown first when half full
static Entry *add(uint64_t obj, uint32_t field) {
    Table *t = g_tab;
    if (!t || 2 * (t->used + 1) > t->mask + 1) {
        uint32_t size = t ? 2 * (t->mask + 1) : 256;
        Table *g = calloc(1, sizeof(Table) + size * sizeof(Entry));
        if (!g) FATAL("percore: out of memory\n");
        g->mask = size - 1;
        for (uint32_t i = 0; t && i <= t->mask; i++)
            if (t->e[i].obj) {
                uint32_t j = slot_of(g, t->e[i].obj, t->e[i].field);
                while (g->e[j].obj) j = (j + 1) & g->mask;
                g->e[j] = t->e[i];
                g->used++;
            }
        __atomic_store_n(&g_tab, g, __ATOMIC_RELEASE);  // (the old table stays: a reader may be in it)
        t = g;
    }
    uint32_t i = slot_of(t, obj, field);
    while (t->e[i].obj) i = (i + 1) & t->mask;
    t->used++;
    return &t->e[i];
}

// The pattern over an array of `bytes`: 8-byte units, then a 4-byte one, then bytes (0xa5) - write it, or the offset
// of the first unit that no longer holds it (-1: all do).
static int64_t poison(uint64_t a, uint32_t bytes, int write) {
    uint32_t o = 0;
    for (; o + 8 <= bytes; o += 8)
        if (write) ST64(a + o, POISON);
        else if (LD64(a + o) != POISON) return o;
    if (o + 4 <= bytes) {
        if (write) ST32(a + o, (uint32_t)POISON);
        else if (LD32(a + o) != (uint32_t)POISON) return o;
        o += 4;
    }
    for (; o < bytes; o++)
        if (write) ST8(a + o, 0xa5);
        else if (LD8(a + o) != 0xa5) return o;
    return -1;
}

static uint64_t block_of(uint64_t obj, uint32_t field, uint32_t stride, uint32_t switch_slots, int with_system, int live);

// The block for obj's array at +field of slots of `stride` bytes, which the Switch sized for `switch_slots` cores
// (with_system: its last slot is the system core's threads').
uint64_t rt_percore_block(uint64_t obj, uint32_t field, uint32_t stride, uint32_t switch_slots, int with_system) {
    return block_of(obj, field, stride, switch_slots, with_system, 0);
}

// The same for an array the game keeps using as well (its slots are roles some code picks by a fixed index): the
// block's first switch_slots slots mirror it (the caller keeps them equal), the array is not poisoned.
uint64_t rt_percore_block_live(uint64_t obj, uint32_t field, uint32_t stride, uint32_t switch_slots) {
    return block_of(obj, field, stride, switch_slots, 0, 1);
}

static uint64_t block_of(uint64_t obj, uint32_t field, uint32_t stride, uint32_t switch_slots, int with_system, int live) {
    Entry *e = find(__atomic_load_n(&g_tab, __ATOMIC_ACQUIRE), obj, field);
    if (e) return e->block;
    pthread_mutex_lock(&g_m);
    e = find(g_tab, obj, field);
    if (e) {
        pthread_mutex_unlock(&g_m);
        return e->block;
    }
    size_t slots = (size_t)rt_game_cores() + (with_system ? 1 : 0);
    void *b = calloc(1, slots * stride + 64);
    if (!b) FATAL("percore: out of memory\n");
    uint64_t block = ((uint64_t)(uintptr_t)b + 63) & ~63ull;
    if (g_poison < 0) g_poison = getenv("SAVAGEWIND_PERCORE_POISON") && atoi(getenv("SAVAGEWIND_PERCORE_POISON"));
    uint32_t bytes = live ? 0 : switch_slots * stride;  // (a live array: nothing poisoned, nothing checked)
    if (g_poison && bytes) poison(obj + field, bytes, 1);
    e = add(obj, field);
    e->block = block, e->field = field, e->bytes = bytes;
    __atomic_store_n(&e->obj, obj, __ATOMIC_RELEASE);
    pthread_mutex_unlock(&g_m);
    LOG("[percore] %llx +%#x: %zu slots of %#x (the game's %u%s)%s\n", (unsigned long long)obj, field, slots, stride,
        switch_slots, with_system ? ", the last the system core's" : "", live ? "; the game's array kept" : g_poison ? "; its own array poisoned" : "");
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
    pthread_mutex_lock(&g_m);
    for (uint32_t i = 0; g_tab && i <= g_tab->mask; i++) {
        Entry *e = &g_tab->e[i];
        if (!e->obj || e->reported) continue;
        int64_t o = poison(e->obj + e->field, e->bytes, 0);
        if (o >= 0) {
            LOG("[percore] MISSED WRITE: %llx +%#x, the game's own array written at +%#llx\n", (unsigned long long)e->obj,
                e->field, (unsigned long long)o);
            e->reported = 1;
        }
    }
    pthread_mutex_unlock(&g_m);
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

// ---- #25: the job manager's per-core scratch (one object): each core's 0x2800-byte scratch buffer (pointers at +0x458;
// a job asking for at most that much gets its core's) and the most any job on the core asked for (u32s at +0x448). The
// game makes, frees and zeroes its three at a time; the constructor's zeroing and the accessors are patched in place
// (tools/patches.py _jobs_*), the init and destructor walk every core here.
#define JOBS_BUFS(obj) rt_percore_block(obj, 0x458, 8, 3, 0)

// The init f_700156da74 (1.9.0), at its first buffer's store: that one is core 0's, then every other core's from the
// same heap (its vtable +0x30: alloc(size, align)).
void rt_jobs_scratch_alloc(Cpu *c, uint64_t obj, uint64_t heap, uint64_t size, uint64_t first) {
    uint64_t b = JOBS_BUFS(obj);
    ST64(b, first);
    for (int i = 1, n = rt_game_cores(); i < n; i++) {
        c->x[0] = heap, c->x[1] = size, c->x[2] = 0x10;
        guest_call(c, LD64(LD64(heap) + 0x30));
        ST64(b + 8ull * (uint64_t)i, c->x[0]);
    }
}

// The destructor f_700156d7b0 (1.9.0): every core's buffer back to the heap (free_fn).
void rt_jobs_scratch_free(Cpu *c, uint64_t obj, uint64_t free_fn) {
    uint64_t b = JOBS_BUFS(obj);
    for (int i = 0, n = rt_game_cores(); i < n; i++) {
        uint64_t p = LD64(b + 8ull * (uint64_t)i);
        if (!p) continue;
        c->x[0] = p;
        guest_call(c, free_fn);
        ST64(b + 8ull * (uint64_t)i, 0);
    }
}

// ---- #17: per-core 0xd50-byte records (count and array at +0x418 / +0x420 of their owner; every user bounds the core
// by the count, the destructor is a plain delete[] by the header's count). The game makes them with a literal 3:
// f_700148e9f4 (1.9.0, holder x0, heap x1) is `new T[3]` with T's constructor inline. This makes the game's core count
// of them. fn: [0] new[](size, heap, align, tag), [1] the tag's GOT slot, [2..5] the parts' constructors: +0;
// +0x68 and +0x188; +0x2a8; eight at +0x3c8 + k x 0x130.
void rt_rec17_new(Cpu *c, const uint64_t *fn) {
    uint64_t holder = c->x[0], heap = c->x[1], n = (uint64_t)rt_game_cores();
    c->x[0] = 8 + n * 0xd50, c->x[1] = heap, c->x[2] = 8, c->x[3] = LD64(fn[1]);
    guest_call(c, fn[0]);
    uint64_t p = c->x[0];
    if (!p) return;
    ST64(p, n);
    for (uint64_t i = 0; i < n; i++) {
        uint64_t e = p + 8 + i * 0xd50;
        c->x[0] = e;
        guest_call(c, fn[2]);
        c->x[0] = e + 0x68;
        guest_call(c, fn[3]);
        c->x[0] = e + 0x188;
        guest_call(c, fn[3]);
        c->x[0] = e + 0x2a8;
        guest_call(c, fn[4]);
        for (uint64_t k = 0; k < 8; k++) {
            c->x[0] = e + 0x3c8 + 0x130 * k;
            guest_call(c, fn[5]);
        }
        ST32(e + 0xd48, 0);
    }
    ST32(holder, n), ST64(holder + 8, p + 8);
    LOG("[percore] #17 records: %llu of 0xd50 at %llx\n", (unsigned long long)n, (unsigned long long)(p + 8));
}

// ---- #7: per-core 0xe8-byte objects (pointers at +8 of their owner; the accessor f_7000ddfea4 returns the current
// core's). The game's constructor f_7000ddfd70 (1.9.0) makes three, its destructor f_7000ddfe04 frees three.
#define OBJ7(obj) rt_percore_block(obj, 8, 8, 3, 0)

// The constructor, for every core: fn: [0] the owner's vtable, [1] the heap getter, [2] new(size, heap, align),
// [3] the object's constructor.
void rt_obj7_make(Cpu *c, const uint64_t *fn) {
    uint64_t self = c->x[0], b = OBJ7(self);
    ST64(self, fn[0]);
    for (int i = 0, n = rt_game_cores(); i < n; i++) {
        guest_call(c, fn[1]);
        c->x[1] = c->x[0], c->x[0] = 0xe8, c->x[2] = 0x2000;
        guest_call(c, fn[2]);
        uint64_t p = c->x[0];
        guest_call(c, fn[3]);  // (x0 = p)
        ST64(b + 8ull * (uint64_t)i, p);
    }
}

// The destructor: every core's object finalized (fini at +0x58) and deleted.
void rt_obj7_free(Cpu *c, uint64_t self, uint64_t fini, uint64_t del) {
    uint64_t b = OBJ7(self);
    for (int i = 0, n = rt_game_cores(); i < n; i++) {
        uint64_t p = LD64(b + 8ull * (uint64_t)i);
        if (p) {
            c->x[0] = p + 0x58;
            guest_call(c, fini);
            c->x[0] = p;
            guest_call(c, del);
        }
        ST64(b + 8ull * (uint64_t)i, 0);
    }
}

// ---- #11: the per-core linear allocators (one object, `*0x7001e377b8`, 0x180 bytes). One 0x58-byte record per core at
// +0x38, its fields (the game's offsets from obj + core x 0x58): +0x38 start and +0x40 cursor in the shared buffer,
// +0x48 its share's size, +0x4c the whole buffer's size, +0x50 / +0x58 / +0x60 a 0x800-byte allocator, +0x68 / +0x70
// and +0x78 / +0x80 two mark buffers (current, start; +0x140 picks which), +0x88 their size, +0x8c a peak. The shared
// buffer (+0x28, size +0x30) is split among the cores every frame. The game's constructor, destructor, init and split
// do three records; these do every core's (the accessors take the block in tools/patches.py _lin_*).
#define LIN(obj) (rt_percore_block(obj, 0x38, 0x58, 3, 0) - 0x38)  // + core x 0x58 + the game's field offset

// f_7000f9d7c8 (1.9.0): the constructor. fn: [0] the vtable, [1] the lock's constructor (+0x148).
void rt_lin_ctor(Cpu *c, const uint64_t *fn) {
    uint64_t obj = c->x[0], r = LIN(obj);
    ST64(obj + 0x28, 0), ST64(obj, fn[0]), ST32(obj + 0x30, 0);
    for (uint64_t i = 0, n = (uint64_t)rt_game_cores(); i < n; i++)
        ST64(r + i * 0x58 + 0x50, 0), ST32(r + i * 0x58 + 0x58, 0), ST64(r + i * 0x58 + 0x60, 0);
    c->x[0] = obj + 0x148;
    guest_call(c, fn[1]);
}

// f_7000f9d80c (1.9.0): the destructor. fn: [0] the vtable, [1] the allocator's finalizer, [2] free, [3] the lock's.
void rt_lin_dtor(Cpu *c, const uint64_t *fn) {
    uint64_t obj = c->x[0], r = LIN(obj);
    ST64(obj + 0x28, 0), ST64(obj, fn[0]);
    for (uint64_t i = 0, n = (uint64_t)rt_game_cores(); i < n; i++) {
        uint64_t ri = r + i * 0x58;
        c->x[0] = ri + 0x50;
        guest_call(c, fn[1]);
        for (int k = 0; k < 2; k++) {
            uint64_t f = ri + 0x68 + 0x10 * (uint64_t)k;
            if (LD64(f)) {
                c->x[0] = LD64(f);
                guest_call(c, fn[2]);
                ST64(f, 0);
            }
        }
    }
    c->x[0] = obj + 0x148;
    guest_call(c, fn[3]);
}

// f_7000f9dac0 (1.9.0): a frame's split (x1: 0 = every core gets the whole buffer, else each its 1/N share, 64-byte
// aligned), and the mark buffers' toggle.
void rt_lin_split(Cpu *c) {
    uint64_t obj = c->x[0], r = LIN(obj), base = LD64(obj + 0x28);
    uint32_t n = (uint32_t)rt_game_cores(), total = LD32(obj + 0x30);
    uint8_t t = (uint8_t)(1 - LD8(obj + 0x140));
    ST8(obj + 0x140, t);
    uint32_t part = (total / n) & 0x7fffffc0u, size = c->x[1] ? part : total;
    for (uint32_t i = 0; i < n; i++) {
        uint64_t ri = r + 0x58ull * i, start = base + (c->x[1] ? (uint64_t)part * i : 0);
        ST64(ri + 0x38, start), ST64(ri + 0x40, start), ST32(ri + 0x48, size);
        ST64(ri + 0x70 + 0x10ull * t, LD64(ri + 0x68 + 0x10ull * t));  // (the toggled mark buffer back to its start)
    }
}

// f_7000f9d8d8 (1.9.0): init(obj, mark buffer size >> 8 (w1), the shared buffer (x2) and its size (w3), heap (x4),
// x5). fn: [0] make an allocator (into x8: {3 qwords}; size, heap, align, 1), [1] its init (allocator, size), [2]
// malloc (size, heap, align), [3] the split, [4] f_7000f74a24 (obj, 1, 2, x5).
void rt_lin_init(Cpu *c, const uint64_t *fn) {
    uint64_t obj = c->x[0], base = c->x[2], heap = c->x[4], x5 = c->x[5];
    uint32_t marks = (uint32_t)c->x[1] << 8, total = (uint32_t)c->x[3];
    ST32(obj + 0x30, total), ST64(obj + 0x28, base);
    uint64_t r = LIN(obj), sp = c->sp, out = (c->sp -= 0x20);
    for (uint64_t i = 0, n = (uint64_t)rt_game_cores(); i < n; i++) {
        uint64_t ri = r + i * 0x58;
        c->x[8] = out, c->x[0] = 0x800, c->x[1] = heap, c->x[2] = 0x40, c->x[3] = 1;
        guest_call(c, fn[0]);
        ST64(ri + 0x50, LD64(out)), ST64(ri + 0x58, LD64(out + 8)), ST64(ri + 0x60, LD64(out + 0x10));
        c->x[0] = ri + 0x50, c->x[1] = 0x800;
        guest_call(c, fn[1]);
        ST32(ri + 0x88, marks), ST32(ri + 0x4c, LD32(obj + 0x30));
        if (!marks) {
            ST64(ri + 0x68, 0), ST64(ri + 0x70, 0), ST64(ri + 0x78, 0), ST64(ri + 0x80, 0);
            continue;
        }
        c->x[0] = marks, c->x[1] = heap, c->x[2] = 0x100;
        guest_call(c, fn[2]);
        ST64(ri + 0x68, c->x[0]), ST64(ri + 0x70, c->x[0]);
        c->x[0] = LD32(ri + 0x88), c->x[1] = heap, c->x[2] = 0x100;
        guest_call(c, fn[2]);
        ST64(ri + 0x78, c->x[0]), ST64(ri + 0x80, c->x[0]);
    }
    c->sp = sp;
    ST8(obj + 0x140, 0);
    c->x[0] = obj, c->x[1] = 0;
    guest_call(c, fn[3]);
    c->x[0] = obj, c->x[1] = 1, c->x[2] = 2, c->x[3] = x5;
    guest_call(c, fn[4]);
}

// ---- #5: per-slot results (slot = sead's core table entry of the core): a 0x40-byte record per slot at +0x1138 (+0x3c
// a valid byte), a u32 per slot at +0x1238, a float per slot at +0x6f0. Jobs fill their slot's; the merge and the
// float pick read every slot's.
#define REC5(obj) rt_percore_block(obj, 0x1138, 0x40, 3, 0)
#define F5(obj) rt_percore_block(obj, 0x6f0, 4, 3, 0)

// f_7000c30e74 (1.9.0): the first valid record (by slot) is the result (its first 0x3c bytes to +0x11f8, +0x1234 = 1);
// every slot's valid byte cleared.
void rt_rec5_merge(Cpu *c) {
    uint64_t obj = c->x[0], b = REC5(obj);
    int n = rt_game_cores();
    for (int i = 0; i < n; i++) {
        uint64_t r = b + 0x40ull * (uint64_t)i;
        if (!LD8(r + 0x3c)) continue;
        for (uint32_t o = 0; o < 0x30; o += 8) ST64(obj + 0x11f8 + o, LD64(r + o));
        ST64(obj + 0x1228, LD64(r + 0x30)), ST32(obj + 0x1230, LD32(r + 0x38)), ST8(obj + 0x1234, 1);
        break;
    }
    for (int i = 0; i < n; i++) ST8(b + 0x40ull * (uint64_t)i + 0x3c, 0);
}

// f_7000c33098 (1.9.0): the slot float with the largest magnitude (the first of equals; a NaN first slot gives 0), in s0.
void rt_f5_pick(Cpu *c) {
    uint64_t b = F5(c->x[0]);
    union { float f; uint32_t u; } v, best;
    v.u = LD32(b);
    float a = v.f > 0.0f ? v.f : -v.f, top = a != a ? -1.0f : (a > -1.0f ? a : -1.0f);
    best.f = a > -1.0f ? v.f : 0.0f;
    for (int i = 1, n = rt_game_cores(); i < n; i++) {
        v.u = LD32(b + 4ull * (uint64_t)i);
        a = v.f > 0.0f ? v.f : -v.f;
        if (a > top) best.f = v.f, top = a;
    }
    c->v[0] = VV(best.u);
}

// ---- #12: per-core 0xfd8-byte objects of the graphics singleton (pointers at +0x70; core-indexed accessors use the
// current core's, other code uses the first three by a fixed index - roles). Its init f_700105bf6c (1.9.0) makes three;
// at its first store this makes every other core's too (the game's second and third then take them).
void rt_obj12_make(Cpu *c, uint64_t obj, uint64_t first, uint64_t heap, uint64_t alloc, uint64_t ctor) {
    uint64_t b = rt_percore_block_live(obj, 0x70, 8, 3);
    ST64(b, first);
    for (uint64_t i = 1, n = (uint64_t)rt_game_cores(); i < n; i++) {
        c->x[0] = 0xfd8, c->x[1] = heap, c->x[2] = 8;
        guest_call(c, alloc);
        uint64_t p = c->x[0];
        c->x[0] = p, c->x[1] = 0;
        guest_call(c, ctor);
        ST64(b + 8 * i, p);
    }
}

// ---- #27: per-slot queues of a few 0x3f0-byte objects (slot = sead's core table entry of the core): a buffer per slot
// (u32 count, pointer) at +0x128, a 0x40-byte lock per slot at +0x328, a cursor per slot at +0x2a0. Jobs append to their
// slot's under its lock (f_7001138a08), a walker drains every slot (f_700113dbdc).

// The constructor's end (f_70011323c0 1.9.0, its tail memset of +0x2a0): every slot's buffer and cursor zeroed, every
// slot's lock made (memset, then the lock's constructor).
void rt_q27_ctor(Cpu *c, uint64_t obj, uint64_t lock_ctor) {
    uint64_t bufs = rt_percore_block(obj, 0x128, 0x10, 3, 0), locks = rt_percore_block(obj, 0x328, 0x40, 3, 0),
             curs = rt_percore_block(obj, 0x2a0, 4, 3, 0);
    for (uint64_t i = 0, n = (uint64_t)rt_game_cores(); i < n; i++) {
        ST32(bufs + 0x10 * i, 0), ST64(bufs + 0x10 * i + 8, 0), ST32(curs + 4 * i, 0);
        for (uint64_t o = 0; o < 0x40; o += 8) ST64(locks + 0x40 * i + o, 0);
        c->x[0] = locks + 0x40 * i;
        guest_call(c, lock_ctor);
    }
}

// The init f_7001134be0 (1.9.0), at its first buffer's store: that one is slot 0's; every other slot's the same size
// (alloc(size, heap, 8, tag)).
void rt_q27_bufs(Cpu *c, uint64_t obj, uint32_t count, uint64_t first, uint64_t heap, uint64_t alloc, uint64_t tag) {
    uint64_t bufs = rt_percore_block(obj, 0x128, 0x10, 3, 0);
    ST32(bufs, count), ST64(bufs + 8, first);
    for (uint64_t i = 1, n = (uint64_t)rt_game_cores(); i < n; i++) {
        c->x[0] = (uint64_t)count * 8, c->x[1] = heap, c->x[2] = 8, c->x[3] = tag;
        guest_call(c, alloc);
        if (c->x[0]) ST32(bufs + 0x10 * i, count), ST64(bufs + 0x10 * i + 8, c->x[0]);
    }
}

// ---- #23: per-core 0x160-byte records of a singleton (`*0x7001e56b20`) at +0x2a0: four 0x58-byte sub-entries each
// (their +0x28 a state, 8 = free). The constructor f_70013ffba0 (1.9.0) builds all twelve of the game's three records the
// same way (a template and constants; nothing in them points into them); at its end the block takes the three it built
// and every further core's a copy of record 0's.
void rt_rec23_ctor_end(uint64_t obj) {
    uint8_t made[3 * 0x160];
    for (uint32_t o = 0; o < sizeof made; o += 8) *(uint64_t *)(made + o) = LD64(obj + 0x2a0 + o);  // (before the block)
    uint64_t b = rt_percore_block(obj, 0x2a0, 0x160, 3, 0);
    for (uint64_t i = 0, n = (uint64_t)rt_game_cores(); i < n; i++) {
        const uint8_t *src = made + 0x160 * (i < 3 ? i : 0);
        for (uint32_t o = 0; o < 0x160; o += 8) ST64(b + 0x160 * i + o, *(const uint64_t *)(src + o));
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
