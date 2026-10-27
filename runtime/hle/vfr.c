// ksys::VFR, BOTW's frame-time singleton, per core the way a PC job system needs it: a slot for every core the game
// runs jobs on, not sead's three (BOTW-JOBS-NATIVE.md). The game's object (0x160 bytes; zeldaret's layout) keeps the
// shared fields - the interval override (+0x28..+0x2c), the new ratio (+0x30), intervals A / B (+0x34 / +0x38), ratios
// A / B (+0x3c / +0x40), the time multipliers (+0x78 count, +0x80 data), the mask (+0x88), interval / frame rate /
// frame time (+0x150..+0x158) - and the game's own code still updates them. The per-core part lives here instead: for
// each core, which buffer it reads (A or B: useBufferA / useBufferB) and both buffers' four deltas (raw delta frame,
// delta frame, raw delta time, delta time). The object's per-core fields (+0x48 / +0x60 and +0xf0..+0x150: six arrays
// of three pointers; +0x8c..+0xec: two 3-core storage blocks) are no longer read: the game's methods that read them
// are replaced at generation time (tools/patches.py, the vfr_* sites), and where the game's own code walks all three
// cores (init, updateInterval, resetTimeMultipliers: copy A to B, spread core 0 to every core) a call here does it for
// every core.
//
// The core: sead's core id (sead::CoreInfo's TLS slot holds core + 1 on job workers), else the current core number,
// as the game's getters read it; the slot variable's address comes from the patch (the core_id_tls site).
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "../hle.h"
#include "../cpu.h"

#define VFR_MAX_CORES 32
#define VFR_INTERVAL_A 0x34
#define VFR_INTERVAL_B 0x38
#define VFR_RATIO_A 0x3c
#define VFR_RATIO_B 0x40
#define VFR_MULTS_N 0x78
#define VFR_MULTS 0x80
#define VFR_FRAME_TIME 0x158

typedef struct {
    float raw_df, df, raw_dt, dt;
} VfrDeltas;
static struct {
    uint8_t buf[VFR_MAX_CORES];          // the buffer each core reads (0: A, 1: B)
    VfrDeltas st[2][VFR_MAX_CORES];      // [buffer][core]
} V;

extern int rt_game_cores(void);
extern uint32_t kernel_current_core(void);

#define F32(a) (*(float *)(uintptr_t)(a))
#define U32(a) (*(uint32_t *)(uintptr_t)(a))

static float arg_s0(Cpu *c) {
    uint32_t u = (uint32_t)c->v[0][0];
    float f;
    memcpy(&f, &u, 4);
    return f;
}
static void ret_s0(Cpu *c, float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    c->v[0] = (VReg){u, 0};
}

// the current core, as the game's getters pick their slot: the ksys TLS value - 1, else the core number; past the
// game's core count, slot 0 (the game's bound)
static uint32_t vfr_core(Cpu *c, uint64_t tls_slot_var) {
    uint32_t slot = U32(tls_slot_var);
    uint64_t thread = *(uint64_t *)(uintptr_t)(c->tpidrro + 0x1f8);
    uint64_t v = *(uint64_t *)(uintptr_t)(thread + 0x80 + 8ull * slot);  // (nn::os::GetTlsValue)
    uint32_t core = v ? (uint32_t)(v - 1) : kernel_current_core();
    return core < (uint32_t)rt_game_cores() ? core : 0;
}
static uint32_t bound(uint32_t core) { return core < (uint32_t)rt_game_cores() ? core : 0; }

static VfrDeltas *cur(uint32_t core) { return &V.st[V.buf[core]][core]; }
static float ratio_of(uint64_t vfr, uint32_t core) { return F32(vfr + (V.buf[core] ? VFR_RATIO_B : VFR_RATIO_A)); }
static uint32_t interval_of(uint64_t vfr, uint32_t core) { return U32(vfr + (V.buf[core] ? VFR_INTERVAL_B : VFR_INTERVAL_A)); }

// setMinDelta(core, mask): the smallest time multiplier the mask selects (1 if none), at least 0.01, as the core's
// deltas
static void set_min_delta(uint64_t vfr, uint32_t core, uint32_t mask) {
    float m = 1.0f;
    uint32_t n = U32(vfr + VFR_MULTS_N);
    uint64_t mults = *(uint64_t *)(uintptr_t)(vfr + VFR_MULTS);
    for (uint32_t i = 0; (int32_t)i < (int32_t)n; i++)  // (as the game: the bit is i mod 32; m stays unless m >= v)
        if (mask >> (i & 31) & 1) {
            float v = F32(mults + 16ull * i + 4);
            m = m < v ? m : v;
        }
    float d = m < 0.01f ? 0.01f : m;
    VfrDeltas *s = cur(core);
    float ft = F32(vfr + VFR_FRAME_TIME);
    s->raw_df = d;
    s->df = d * ratio_of(vfr, core);
    s->raw_dt = d * ft;
    s->dt = s->df * ft;
}

// ---- the game's methods (each patch passes its Cpu and the ksys TLS slot variable)
void rt_vfr_delta_frame(Cpu *c, uint64_t tls) { ret_s0(c, cur(vfr_core(c, tls))->df); }
void rt_vfr_raw_delta_frame(Cpu *c, uint64_t tls) { ret_s0(c, cur(vfr_core(c, tls))->raw_df); }
void rt_vfr_delta_time(Cpu *c, uint64_t tls) { ret_s0(c, cur(vfr_core(c, tls))->dt); }
void rt_vfr_interval(Cpu *c, uint64_t tls) {
    uint32_t v = interval_of(c->x[0], vfr_core(c, tls));
    c->x[0] = v > 1 ? v : 1;
}
void rt_vfr_ratio(Cpu *c, uint64_t tls) { ret_s0(c, ratio_of(c->x[0], vfr_core(c, tls))); }
void rt_vfr_ratio_time(Cpu *c, uint64_t tls) { ret_s0(c, ratio_of(c->x[0], vfr_core(c, tls)) * F32(c->x[0] + VFR_FRAME_TIME)); }
void rt_vfr_use_a(Cpu *c, uint64_t tls) { V.buf[vfr_core(c, tls)] = 0; }
void rt_vfr_use_b(Cpu *c, uint64_t tls) { V.buf[vfr_core(c, tls)] = 1; }
// setMinDelta(this, core, const BitFlag32 *mask)
void rt_vfr_set_min_delta(Cpu *c, uint64_t tls) {
    (void)tls;
    set_min_delta(c->x[0], bound((uint32_t)c->x[1]), U32(c->x[2]));
}
// getDeltaAndSetMin(this, f32 *raw_out, include, exclude): the current core's delta frame (returned) and raw delta
// frame (stored) before its delta is set from (mask | include) & ~exclude
void rt_vfr_get_and_set_min(Cpu *c, uint64_t tls) {
    uint64_t vfr = c->x[0];
    uint32_t core = vfr_core(c, tls);
    VfrDeltas *s = cur(core);
    F32(c->x[1]) = s->raw_df;
    float d = s->df;
    set_min_delta(vfr, core, (U32(vfr + 0x88) | (uint32_t)c->x[2]) & ~(uint32_t)c->x[3]);
    ret_s0(c, d);
}
// setDelta(this, f32): the current core's deltas from d (at least 0.01)
void rt_vfr_set_delta(Cpu *c, uint64_t tls) {
    uint64_t vfr = c->x[0];
    uint32_t core = vfr_core(c, tls);
    float d = arg_s0(c);
    d = d < 0.01f ? 0.01f : d;
    VfrDeltas *s = cur(core);
    float ft = F32(vfr + VFR_FRAME_TIME);
    s->raw_df = d;
    s->df = d * ratio_of(vfr, core);
    s->raw_dt = d * ft;
    s->dt = s->df * ft;
}

// ---- where the game's own code walks all three cores
// buffer A's deltas to buffer B, every core (init, updateInterval; the game copies the shared interval and ratio)
void rt_vfr_copy_a_to_b(Cpu *c) {
    (void)c;
    memcpy(V.st[1], V.st[0], sizeof V.st[0]);
}
// core 0's buffer A deltas to every core (setMinDelta(): init, updateInterval, resetTimeMultipliers)
void rt_vfr_spread_core0(Cpu *c) {
    (void)c;
    for (int i = 1; i < VFR_MAX_CORES; i++) V.st[0][i] = V.st[0][0];
}
// init: both (after its setMinDelta(0)). (check) SAVAGEWIND_VFR_POISON=1: the object's six per-core pointer arrays,
// which nothing should read any more, get null-page addresses (0x100 + their offset): a game function still reading
// one faults at that read, with its guest address in the crash report, instead of running on stale deltas.
void rt_vfr_init_spread(Cpu *c) {
    rt_vfr_spread_core0(c);
    rt_vfr_copy_a_to_b(c);
    const char *e = getenv("SAVAGEWIND_VFR_POISON");
    if (e && *e == '1') {
        static const uint32_t arrays[] = {0x48, 0x60, 0xf0, 0x108, 0x120, 0x138};
        for (int a = 0; a < 6; a++)
            for (int i = 0; i < 3; i++) *(uint64_t *)(uintptr_t)(c->x[19] + arrays[a] + 8 * i) = 0x100 + arrays[a];
        LOG("[vfr] per-core pointer arrays of %llx poisoned\n", (unsigned long long)c->x[19]);
    }
}

// (runtime/hle/framepace.c) core 0's delta frame in buffer A: the frame that just ended
float rt_vfr_core0_delta_frame(void) { return V.st[0][0].df; }
float rt_vfr_core_delta_frame(int core) { return core >= 0 && core < VFR_MAX_CORES ? cur((uint32_t)core)->df : 0; }
