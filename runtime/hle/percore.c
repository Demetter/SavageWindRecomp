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
    uint32_t span, stride;  // (the block's slots: span bytes; the guard after them)
    uint8_t reported;       // (check: a missed write logged)
    uint8_t overrun;        // (guard: a write past the slots logged)
} Entry;
#define GUARD 64
#define GUARD_BYTE 0xfd
static int g_guard = -1;  // (check) SAVAGEWIND_PERCORE_GUARD=1: GUARD bytes after every block's slots, checked each frame
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
    if (g_guard < 0) g_guard = getenv("SAVAGEWIND_PERCORE_GUARD") && atoi(getenv("SAVAGEWIND_PERCORE_GUARD"));
    void *b = calloc(1, slots * stride + 64 + GUARD);
    if (!b) FATAL("percore: out of memory\n");
    uint64_t block = ((uint64_t)(uintptr_t)b + 63) & ~63ull;
    if (g_guard) memset((void *)(uintptr_t)(block + slots * stride), GUARD_BYTE, GUARD);
    if (g_poison < 0) g_poison = getenv("SAVAGEWIND_PERCORE_POISON") && atoi(getenv("SAVAGEWIND_PERCORE_POISON"));
    uint32_t bytes = live ? 0 : switch_slots * stride;  // (a live array: nothing poisoned, nothing checked)
    if (g_poison && bytes) poison(obj + field, bytes, 1);
    e = add(obj, field);
    e->block = block, e->field = field, e->bytes = bytes, e->span = (uint32_t)(slots * stride), e->stride = stride;
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

// (check) once a frame (runtime/hle/framepace.c): with the guard, every block's guard intact (a write past its slots:
// a core the game indexes beyond the block, which would corrupt the host heap); every ~600 frames, the poisoned
// arrays still poisoned?
void rt_percore_check(void) {
    static uint64_t frames;
    if (g_guard > 0) {
        pthread_mutex_lock(&g_m);
        for (uint32_t i = 0; g_tab && i <= g_tab->mask; i++) {
            Entry *e = &g_tab->e[i];
            if (!e->obj || e->overrun) continue;
            const uint8_t *g = (const uint8_t *)(uintptr_t)(e->block + e->span);
            for (int k = 0; k < GUARD; k++)
                if (g[k] != GUARD_BYTE) {
                    LOG("[percore] OVERRUN: %llx +%#x (%u-byte slots, %u slots): written %d bytes past the last slot\n",
                        (unsigned long long)e->obj, e->field, e->stride, e->span / (e->stride ? e->stride : 1), k);
                    e->overrun = 1;
                    break;
                }
        }
        pthread_mutex_unlock(&g_m);
    }
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
void cmdbuf_arena_region(uint64_t pool, uint64_t offset, uint64_t size);  // (runtime/nvn/cmdbuf.c)

extern uint64_t g_cmd_arena;  // (runtime/hle/jobs.c rt_view_rec_inline)

void rt_arena_setup(Cpu *c, const uint64_t *fn) {
    uint64_t obj = c->x[0], ctl = c->x[2], heap = c->x[5], blk = (c->x[1] + 3) & ~3ull;
    g_cmd_arena = obj;
    uint32_t nbuf = (uint32_t)((uint64_t)(uint32_t)c->x[3] * (uint64_t)rt_game_cores() / 3);  // (command buffers a frame
    // for every recorder: the Switch build's count is for three cores' worth of parts, the views' recording among them)
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
                cmdbuf_arena_region(LD64(out), LD32(out + 8), blk);  // (this core's thread records there: its own
                // region of command memory, runtime/nvn/cmdbuf.c)
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

// ---- #4a: per-slot 0x70-byte candidates (slot = sead's core table entry of the core) at +0xe00 of the #5 object -
// +0x5a0 of its part at +0x860, which makes them (f_7000c31f04 1.9.0) and releases them (f_7000c2ae3c): a string, a
// priority (+0x58) and a handle (+0x60). Jobs offer a candidate to their slot (kept when its priority is higher), once
// a frame the best is copied to +0xd90 and every slot reset. The game's code does this for three; these for every slot.
#define REC4A(obj) rt_percore_block(obj, 0xe00, 0x70, 3, 0)

// At the call for slot 2 (the record's constructor, its reset, its handle's release at +0x60): that call for slot 2 and
// every further slot (down: from the last).
void rt_rec4a_rest(Cpu *c, uint64_t obj, uint64_t fn, uint32_t at, int down) {
    uint64_t b = REC4A(obj), n = (uint64_t)rt_game_cores();
    for (uint64_t j = 2; j < n; j++) {
        c->x[0] = b + 0x70 * (down ? n + 1 - j : j) + at;
        guest_call(c, fn);
    }
}

// f_7000c2d894 (1.9.0): the slot with the highest priority (the first of equals) copied to +0xd90, every slot reset.
void rt_rec4a_pick(Cpu *c, uint64_t copy, uint64_t reset) {
    uint64_t obj = c->x[0], b = REC4A(obj);
    uint64_t n = (uint64_t)rt_game_cores(), best = 0;
    int32_t top = (int32_t)LD32(b + 0x58);
    for (uint64_t i = 1; i < n; i++) {
        int32_t p = (int32_t)LD32(b + 0x70 * i + 0x58);
        if (top < p) top = p, best = i;
    }
    c->x[0] = obj + 0xd90, c->x[1] = b + 0x70 * best;
    guest_call(c, copy);
    for (uint64_t i = 0; i < n; i++) {
        c->x[0] = b + 0x70 * i;
        guest_call(c, reset);
    }
}

// ---- #4b: per-slot candidate lists (slot = sead's core table entry of the core) of the object f_70010d4924 (1.9.0)
// makes: eight 0x50-byte lists per slot at +0x28 (one per kind; count, capacity 8, pointer to its own buffer at +0x10),
// one list per slot at +0xa28 (A) and at +0xb18 (B). Jobs add their candidates to their slot's lists (f_70010d5534);
// once a frame (f_70010d61d4) the eight merged lists at +0x7a8 are picked from every slot's (best priority, +0x14 of an
// item), dead candidates dropped, the lists cleared. The game's code for this is unrolled for three slots; these are
// the same over every slot.
#define L4B_R(obj) rt_percore_block(obj, 0x28, 0x280, 3, 0)
#define L4B_A(obj) rt_percore_block(obj, 0xa28, 0x50, 3, 0)
#define L4B_B(obj) rt_percore_block(obj, 0xb18, 0x50, 3, 0)

static uint64_t l4b_at(uint64_t l, uint32_t i) { return LD64(LD64(l + 8) + 8ull * i); }

// The index of item in the list, or -1.
static int64_t l4b_find(uint64_t l, uint64_t item) {
    for (int32_t i = 0, n = (int32_t)LD32(l); i < n; i++)
        if (l4b_at(l, (uint32_t)i) == item) return i;
    return -1;
}

// f_7000e8cc5c (1.9.0) for one entry: the entry at i out, the rest moved down.
static void l4b_erase(uint64_t l, uint32_t i) {
    int32_t n = (int32_t)LD32(l), end = (int32_t)i + 1;
    if (n < end) return;
    if (n > end) {
        uint64_t p = LD64(l + 8);
        memmove((void *)(uintptr_t)(p + 8ull * i), (const void *)(uintptr_t)(p + 8ull * (uint32_t)end), 8ull * (uint32_t)(n - end));
    }
    ST32(l, (uint32_t)(n - 1));
}

// An item's kind (0..7: its list) and type, through its descriptor (f_70010d389c / f_70010d38ac).
static uint32_t l4b_kind(uint64_t item) { return LD32(LD64(LD64(item + 8) + 0xa0) + 0x2b0); }
static uint32_t l4b_type(uint64_t item) { return LD32(LD64(LD64(item + 8) + 0xa0) + 0x2b4); }

static float l4b_prio(uint64_t item) {
    union { float f; uint32_t u; } v;
    v.u = LD32(item + 0x14);
    return v.f;
}

static void l4b_init(uint64_t l) {  // (memset, then f_7000e8cb64(l, 8, l + 0x10))
    for (uint32_t o = 0; o < 0x50; o += 8) ST64(l + o, 0);
    ST64(l + 8, l + 0x10), ST32(l, 0), ST32(l + 4, 8);
}

// The constructor's end (its last store): every slot's lists made in the blocks.
void rt_rec4b_ctor(uint64_t obj) {
    uint64_t r = L4B_R(obj), a = L4B_A(obj), b = L4B_B(obj);
    for (uint64_t s = 0, n = (uint64_t)rt_game_cores(); s < n; s++) {
        for (uint64_t k = 0; k < 8; k++) l4b_init(r + 0x280 * s + 0x50 * k);
        l4b_init(a + 0x50 * s), l4b_init(b + 0x50 * s);
    }
}

// f_70010d5fe8 (1.9.0): every list emptied (every slot's, the merged ones), the current pick and the state cleared.
void rt_rec4b_reset(Cpu *c) {
    uint64_t obj = c->x[0], r = L4B_R(obj), a = L4B_A(obj), b = L4B_B(obj);
    for (uint64_t s = 0, n = (uint64_t)rt_game_cores(); s < n; s++) {
        for (uint64_t k = 0; k < 8; k++) ST32(r + 0x280 * s + 0x50 * k, 0);
        ST32(a + 0x50 * s, 0), ST32(b + 0x50 * s, 0);
    }
    for (uint64_t k = 0; k < 8; k++) ST32(obj + 0x7a8 + 0x50 * k, 0);
    ST64(obj + 0xc08, 0), ST64(obj + 0xc40, 0), ST16(obj + 0xe20, 0);
}

// The per-frame update's clearing of every slot's lists (the kinds' and A's; f_70010d61d4 1.9.0, two places).
void rt_rec4b_clear(uint64_t obj) {
    uint64_t r = L4B_R(obj), a = L4B_A(obj);
    for (uint64_t s = 0, n = (uint64_t)rt_game_cores(); s < n; s++) {
        for (uint64_t k = 0; k < 8; k++) ST32(r + 0x280 * s + 0x50 * k, 0);
        ST32(a + 0x50 * s, 0);
    }
}

// Does any slot have a candidate of kind 1? (f_70010d61d4: its three counts read one after another)
int rt_rec4b_any1(uint64_t obj) {
    uint64_t r = L4B_R(obj);
    for (uint64_t s = 0, n = (uint64_t)rt_game_cores(); s < n; s++)
        if (LD32(r + 0x280 * s + 0x50)) return 1;
    return 0;
}

// Does any slot's kind-1 list hold a candidate other than `cur`? (f_70010d61d4: three unrolled scans)
int rt_rec4b_other1(uint64_t obj, uint64_t cur) {
    uint64_t r = L4B_R(obj);
    for (uint64_t s = 0, n = (uint64_t)rt_game_cores(); s < n; s++) {
        uint64_t l = r + 0x280 * s + 0x50;
        for (int32_t i = 0, m = (int32_t)LD32(l); i < m; i++)
            if (l4b_at(l, (uint32_t)i) != cur) return 1;
    }
    return 0;
}

// f_70010d61d4's loop over the merged kind-1 list (+0x7f8), from its end: a candidate no slot's A list holds is dropped.
void rt_rec4b_keep_a(uint64_t obj, uint64_t count) {
    uint64_t a = L4B_A(obj), m = obj + 0x7f8;
    int n = rt_game_cores();
    for (uint64_t i = count; i-- > 0;) {
        uint64_t item = i < LD32(m) ? l4b_at(m, (uint32_t)i) : 0;
        int held = 0;
        for (int s = 0; s < n && !held; s++) held = l4b_find(a + 0x50ull * (uint64_t)s, item) >= 0;
        if (!held) l4b_erase(m, (uint32_t)i);
    }
}

// f_70010d58c4 (1.9.0): the candidate out of every slot's list of its kind (every time it is there).
void rt_rec4b_remove(Cpu *c) {
    uint64_t obj = c->x[0], item = c->x[1], r = L4B_R(obj);
    uint32_t k = l4b_kind(item);
    k = k < 8 ? k : 0;
    for (uint64_t s = 0, n = (uint64_t)rt_game_cores(); s < n; s++) {
        uint64_t l = r + 0x280 * s + 0x50ull * k;
        for (int64_t i; (i = l4b_find(l, item)) >= 0;) l4b_erase(l, (uint32_t)i);
    }
}

// f_70010d5b68 / f_70010d5c40 (1.9.0): the candidate out of every slot's A (B) list (its first place in each).
static void remove_first(uint64_t base, uint64_t item) {
    for (uint64_t s = 0, n = (uint64_t)rt_game_cores(); s < n; s++) {
        int64_t i = l4b_find(base + 0x50 * s, item);
        if (i >= 0) l4b_erase(base + 0x50 * s, (uint32_t)i);
    }
}
void rt_rec4b_remove_a(Cpu *c) { remove_first(L4B_A(c->x[0]), c->x[1]); }
void rt_rec4b_remove_b(Cpu *c) { remove_first(L4B_B(c->x[0]), c->x[1]); }

// A candidate whose owner (its first field) is gone, dying (+0x64 == 3) or flagged (+0x78 bit 0), or which is itself
// flagged (+0x61).
static int l4b_dead(uint64_t item) {
    uint64_t o = LD64(item);
    return !o || LD8(o + 0x64) == 3 || (LD32(o + 0x78) & 1) || LD8(item + 0x61);
}

static void l4b_sweep(uint64_t l) {
    int32_t n = (int32_t)LD32(l);
    for (int32_t i = n - 1; i >= 0; i--)
        if (l4b_dead(l4b_at(l, (uint32_t)i))) l4b_erase(l, (uint32_t)i);
}

// f_70010d6a24 (1.9.0): dead candidates out of every slot's lists; a dead one in a merged list out of everything
// (f_70010d5d18).
void rt_rec4b_sweep(Cpu *c, uint64_t drop) {
    uint64_t obj = c->x[0], r = L4B_R(obj), a = L4B_A(obj), b = L4B_B(obj);
    uint64_t n = (uint64_t)rt_game_cores();
    for (uint64_t s = 0; s < n; s++)
        for (uint64_t k = 0; k < 8; k++) l4b_sweep(r + 0x280 * s + 0x50 * k);
    for (uint64_t s = 0; s < n; s++) l4b_sweep(a + 0x50 * s);
    for (uint64_t s = 0; s < n; s++) l4b_sweep(b + 0x50 * s);
    for (uint64_t k = 0; k < 8; k++) {
        uint64_t m = obj + 0x7a8 + 0x50 * k;
        for (int32_t i = (int32_t)LD32(m) - 1; i >= 0; i--) {
            if (!l4b_dead(l4b_at(m, (uint32_t)i))) continue;
            c->x[0] = obj, c->x[1] = (uint32_t)i < LD32(m) ? l4b_at(m, (uint32_t)i) : 0;
            guest_call(c, drop);
        }
    }
}

// The first candidate of a list not already in the merged list m: the ones before it dropped (f_70010d6f60 /
// f_70010d7930 1.9.0). 0: none left.
static uint64_t l4b_front(uint64_t l, uint64_t m) {
    while (LD32(l)) {
        uint64_t item = l4b_at(l, 0);
        if (l4b_find(m, item) < 0) return item;
        l4b_erase(l, 0);
    }
    return 0;
}

// f_70010d6f60 (1.9.0): the next candidate of a kind - of the slots' first ones not merged yet, the best priority (ties:
// the first slot's, unless a later one is `cur`) - taken out of its slot's list. 0: none.
void rt_rec4b_pick(Cpu *c) {
    uint64_t obj = c->x[0], cur = c->x[2], r = L4B_R(obj);
    uint32_t k = (uint32_t)c->x[1];
    uint64_t m = obj + 0x7a8 + 0x50ull * k;
    float best = -1.0f;
    int64_t slot = -1;
    for (uint64_t s = 0, n = (uint64_t)rt_game_cores(); s < n; s++) {
        uint64_t item = l4b_front(r + 0x280 * s + 0x50ull * k, m);
        if (!item) continue;
        float p = l4b_prio(item);
        if (!(p < best) && (p != best || item == cur)) best = p, slot = (int64_t)s;
    }
    uint64_t l = r + 0x280 * (uint64_t)slot + 0x50ull * k, item = 0;
    if (slot >= 0 && (int32_t)LD32(l) >= 1) item = l4b_at(l, 0), l4b_erase(l, 0);
    c->x[0] = item;
}

// f_70010d7930 (1.9.0): the same over every slot's B list (against the merged kind-1 list; ties: the first slot's).
void rt_rec4b_pick_b(Cpu *c) {
    uint64_t obj = c->x[0], b = L4B_B(obj), m = obj + 0x7f8;
    float best = -1.0f;
    int64_t slot = -1;
    for (uint64_t s = 0, n = (uint64_t)rt_game_cores(); s < n; s++) {
        uint64_t item = l4b_front(b + 0x50 * s, m);
        if (item && l4b_prio(item) > best) best = l4b_prio(item), slot = (int64_t)s;
    }
    uint64_t l = b + 0x50 * (uint64_t)slot, item = 0;
    if (slot >= 0 && (int32_t)LD32(l) >= 1) item = l4b_at(l, 0), l4b_erase(l, 0);
    c->x[0] = item;
}

// f_70010d71ec (1.9.0): each slot's B list made from its kind-1 list (its first eight, as far as B has room).
void rt_rec4b_fill_b(Cpu *c) {
    uint64_t obj = c->x[0], r = L4B_R(obj), b = L4B_B(obj);
    uint64_t n = (uint64_t)rt_game_cores();
    for (uint64_t s = 0; s < n; s++) ST32(b + 0x50 * s, 0);
    for (uint64_t s = 0; s < n; s++) {
        uint64_t src = r + 0x280 * s + 0x50, dst = b + 0x50 * s;
        int32_t m = (int32_t)LD32(src) < 8 ? (int32_t)LD32(src) : 8;
        for (int32_t i = 0; i < m; i++) {
            int32_t cnt = (int32_t)LD32(dst);
            if (cnt >= (int32_t)LD32(dst + 4)) continue;
            ST64(LD64(dst + 8) + 8ull * (uint32_t)cnt, l4b_at(src, (uint32_t)i));
            ST32(dst, (uint32_t)(cnt + 1));
        }
    }
}

// f_70010d7374 (1.9.0): when a kind-0 candidate of the types 0x1800006..8 or 0x180000a has an owner the object's
// filter (+0xc28) takes, every such candidate the filter doesn't take is dropped from every slot's kind-0 list.
static int l4b_filtered(uint32_t type) {
    uint32_t t = type - 0x1800006u;
    return t <= 4 && t != 3;
}
static int l4b_takes(Cpu *c, uint64_t obj, uint64_t item) {
    uint64_t f = LD64(obj + 0xc28);
    if (!f) return 0;
    c->x[0] = LD64(item);
    guest_call(c, f);
    return c->x[0] & 1;
}
void rt_rec4b_filter(Cpu *c) {
    uint64_t obj = c->x[0], r = L4B_R(obj);
    uint64_t n = (uint64_t)rt_game_cores();
    int any = 0;
    for (uint64_t s = 0; s < n && !any; s++) {
        uint64_t l = r + 0x280 * s, p = LD64(l + 8);
        for (int32_t i = 0, m = (int32_t)LD32(l); i < m && !any; i++) {
            uint64_t item = LD64(p + 8ull * (uint32_t)i);
            any = l4b_filtered(l4b_type(item)) && l4b_takes(c, obj, item);
        }
    }
    if (!any) return;
    for (uint64_t s = 0; s < n; s++) {
        uint64_t l = r + 0x280 * s;
        for (int32_t i = (int32_t)LD32(l) - 1; i >= 0; i--) {
            uint64_t item = (uint32_t)i < LD32(l) ? l4b_at(l, (uint32_t)i) : 0;
            if (l4b_filtered(l4b_type(item)) && !l4b_takes(c, obj, item)) l4b_erase(l, (uint32_t)i);
        }
    }
}

// f_70010d7610 (1.9.0): by the states (+0x10) of the kind-0 candidates: when one is 6, the 5s and 4s are dropped from
// every slot's kind-0 list; else when one is 5, the 4s.
static int l4b_has_state(uint64_t r, uint64_t n, uint32_t v) {
    for (uint64_t s = 0; s < n; s++) {
        uint64_t l = r + 0x280 * s;
        for (int32_t i = 0, m = (int32_t)LD32(l); i < m; i++)
            if (LD32(l4b_at(l, (uint32_t)i) + 0x10) == v) return 1;
    }
    return 0;
}
void rt_rec4b_states(Cpu *c) {
    uint64_t obj = c->x[0], r = L4B_R(obj);
    uint64_t n = (uint64_t)rt_game_cores();
    int drop5 = l4b_has_state(r, n, 6);
    if (!drop5 && !l4b_has_state(r, n, 5)) return;
    for (uint64_t s = 0; s < n; s++) {
        uint64_t l = r + 0x280 * s;
        for (int32_t i = (int32_t)LD32(l) - 1; i >= 0; i--) {
            uint32_t v = LD32(l4b_at(l, (uint32_t)i) + 0x10);
            if (v == 4 || (drop5 && v == 5)) l4b_erase(l, (uint32_t)i);
        }
    }
}

// ---- #4c: per-slot rings (slot = sead's core table entry of the core) of the 0x3098-byte object at +0xbf0 of its
// owner (made by f_700112b6a8 1.9.0): two groups, at +8 and at +0xc50, of 0x418-byte rings - a pointer to its own 64
// 16-byte entries at +0x14, the capacity, the start (+0xc) and the count (+0x10). Jobs push to their slot's ring of a
// group (f_7001150e2c); f_7001150fcc drains each group, the first non-empty ring first, into the object's merged
// arrays. The game's code does this for three rings a group; these for one a slot (a block per group, keyed by it).
#define RING4C(group) rt_percore_block(group, 0, 0x418, 3, 0)

// The constructor's end: every slot's ring of both groups, made as the constructor makes the game's (empty entries:
// their +8 all ones).
void rt_rec4c_ctor(uint64_t obj) {
    for (uint64_t g = obj + 8; g <= obj + 0xc50; g += 0xc48) {
        uint64_t b = RING4C(g);
        for (uint64_t s = 0, n = (uint64_t)rt_game_cores(); s < n; s++) {
            uint64_t r = b + 0x418 * s;
            for (uint32_t o = 0; o < 0x418; o += 8) ST64(r + o, 0);
            for (uint32_t e = 0; e < 0x40; e++) ST32(r + 0x14 + 0x10 * e + 8, 0xffffffffu);
            ST64(r, r + 0x14), ST32(r + 8, 0x40);
        }
    }
}

// f_7001150ed4 / f_7001150f50 (1.9.0): the oldest entry of the group's first non-empty ring (by slot) to x1; w0: one
// was there.
void rt_rec4c_pop(Cpu *c) {
    uint64_t b = RING4C(c->x[0]), out = c->x[1];
    for (uint64_t s = 0, n = (uint64_t)rt_game_cores(); s < n; s++) {
        uint64_t r = b + 0x418 * s;
        int32_t cnt = (int32_t)LD32(r + 0x10);
        if (!cnt) continue;
        if (cnt >= 1) {
            int32_t cap = (int32_t)LD32(r + 8), st = (int32_t)LD32(r + 0xc);
            uint64_t e = LD64(r) + 16 * (int64_t)(st - (st < cap ? 0 : cap));
            ST64(out, LD64(e)), ST64(out + 8, LD64(e + 8));
            ST32(r + 0xc, (uint32_t)(st + 1 >= cap ? 0 : st + 1)), ST32(r + 0x10, (uint32_t)(cnt - 1));
        }
        c->x[0] = 1;
        return;
    }
    c->x[0] = 0;
}

// Is any slot's ring of the group not empty? (f_7001150fcc: its three counts read one after another)
int rt_rec4c_any(uint64_t group) {
    uint64_t b = RING4C(group);
    for (uint64_t s = 0, n = (uint64_t)rt_game_cores(); s < n; s++)
        if (LD32(b + 0x418 * s + 0x10)) return 1;
    return 0;
}

// ---- #19: per-core rings - 0x28-byte objects at +0x348 / +0x350 / +0x358 of their owner (made by f_70010605dc 1.9.0):
// the count of cores at +8 (the makers' literal 3), the capacity per core at +0xc, the entries at +0x10 laid out
// [cursor][core], a cursor per core at +0x18 and the capacity again at +0x24. Each core appends at its own cursor; the
// walkers go over every core's entries. The makers size the entries and set the count for every core (tools/patches.py);
// the cursors live in a block per ring.
#define CUR19(ring) rt_percore_block(ring, 0x18, 4, 3, 0)

// A ring's cursors, all zeroed (its maker's, the owner's reset).
void rt_ring19_zero(uint64_t ring) {
    uint64_t b = CUR19(ring);
    for (uint64_t i = 0, n = (uint64_t)rt_game_cores(); i < n; i++) ST32(b + 4 * i, 0);
}

// f_7001060774 (1.9.0): every ring's cursors zeroed.
void rt_ring19_reset(Cpu *c) {
    for (uint32_t o = 0x348; o <= 0x358; o += 8) rt_ring19_zero(LD64(c->x[0] + o));
}

// Ring C's maker f_7001060a0c (1.9.0) constructs its 24 entries unrolled; at the last call: that entry's and every
// further core's (8 a core), each made and its +0x74 zeroed (the maker zeroes the 24th's after the call).
void rt_ring19_c_rest(Cpu *c, uint64_t entries, uint64_t make) {
    for (uint64_t e = 23, n = 8 * (uint64_t)rt_game_cores(); e < n; e++) {
        c->x[0] = entries + 0x7c * e;
        guest_call(c, make);
        if (e != 23) ST32(entries + 0x7c * e + 0x74, 0);
    }
}

// ---- #28's second table: per-core records of the object f_70010b6b94 (1.9.0) makes - a 16-byte interface per core at
// +0x140 and a 0x58-byte polymorphic record per core at +0x1a0 (made with its interface), six of each; f_70010b7184 picks
// the current core's (past the sixth: the first). Blocks with a slot for every core and the system core's threads'.
#define IFACE28(obj) rt_percore_block(obj, 0x140, 0x10, 6, 1)
#define REC28(obj) rt_percore_block(obj, 0x1a0, 0x58, 6, 1)

// The constructor's interfaces (each two vtable pointers): every slot's.
void rt_obj28_ifaces(uint64_t obj, uint64_t a, uint64_t b) {
    uint64_t f = IFACE28(obj);
    for (uint64_t i = 0, n = (uint64_t)rt_game_cores() + 1; i < n; i++) ST64(f + 0x10 * i, a), ST64(f + 0x10 * i + 8, b);
}

// The constructor's first record (make(record, its interface)): every slot's.
void rt_obj28_make(Cpu *c, uint64_t obj, uint64_t make) {
    uint64_t f = IFACE28(obj), r = REC28(obj);
    for (uint64_t i = 0, n = (uint64_t)rt_game_cores() + 1; i < n; i++) {
        c->x[0] = r + 0x58 * i, c->x[1] = f + 0x10 * i;
        guest_call(c, make);
    }
}

// The destructor's first record (the game's last): every slot's, the last first.
void rt_obj28_destroy(Cpu *c, uint64_t obj, uint64_t destroy) {
    uint64_t r = REC28(obj);
    for (uint64_t i = (uint64_t)rt_game_cores() + 1; i-- > 0;) {
        c->x[0] = r + 0x58 * i;
        guest_call(c, destroy);
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
