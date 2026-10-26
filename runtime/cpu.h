// Guest CPU state and helpers used by recompiled code.
#pragma once
#include <stddef.h>
#include <stdint.h>
#include <string.h>

// A V register: a vector, so code that keeps it in a local or forwards it between ops uses a vector register (as a
// 128-bit integer it went through pairs of integer registers). simd.h VV / VU convert from and to 128-bit integers.
typedef uint64_t VReg __attribute__((vector_size(16)));

typedef struct __attribute__((aligned(16))) Cpu {
    uint64_t x[32];          // x0-x30 (x[31] unused; sp is separate)
    uint64_t sp;             // 256
    uint64_t pc;             // 264: target for indirect dispatch
    uint64_t nzcv;           // 272: flags in bits 31..28
    uint64_t tpidr;          // 280
    uint64_t tpidrro;        // 288
    uint64_t excl_addr;      // 296
    uint64_t excl_val;       // 304
    void *thread;            // 312: runtime thread object
    uint64_t fpcr;           // 320: guest FPCR/FPSR (portable FP/SIMD lowering only; the asm path uses the host's)
    uint64_t fpsr;           // 328
    // (runtime/kernel.c) the best priority waiting for the emulated core this thread holds, or NULL (none held, or
    // SAVAGEWIND_CORES=0), and the thread's priority: a call site's inline fast path checks them (RT_PREEMPT_DUE)
    const volatile int *best_wait;  // 336
    int32_t prio;                   // 344
    uint8_t _pad[512 - 348];
    VReg v[32];              // 512
} Cpu;

_Static_assert(offsetof(Cpu, sp) == 256, "sp");
_Static_assert(offsetof(Cpu, pc) == 264, "pc");
_Static_assert(offsetof(Cpu, nzcv) == 272, "nzcv");
_Static_assert(offsetof(Cpu, fpcr) == 320, "fpcr");
_Static_assert(offsetof(Cpu, fpsr) == 328, "fpsr");
_Static_assert(offsetof(Cpu, v) == 512, "v");

typedef void (*GuestFn)(Cpu *);

// ---- runtime services (runtime/*.c)
void rt_dispatch(Cpu *c);                 // call function at c->pc
// An indirect call site's own cache (generated code: blr / br): an entry for one target, never freed or changed, so a
// site keeps a pointer to it and the call happens at the site (a branch of its own for the CPU's predictor).
typedef struct DispatchEnt { uint64_t pc; GuestFn fn; } DispatchEnt;
GuestFn rt_dispatch_site(Cpu *c, const DispatchEnt **site);  // the function at c->pc, for the caller to call
// A better thread waits for this thread's emulated core: an indirect call goes through rt_dispatch_site, whose
// kernel_preempt hands it over (the generated code calls a site's cached target inline otherwise: tools/gen.py)
#define RT_PREEMPT_DUE(c) ((c)->best_wait && *(c)->best_wait < (c)->prio)
void rt_svc(Cpu *c, uint32_t imm, uint64_t pc);
void rt_brk(Cpu *c, uint32_t imm, uint64_t pc) __attribute__((noreturn));
void rt_udf(Cpu *c, uint64_t pc) __attribute__((noreturn));
uint64_t rt_ticks(void);

#if defined(__aarch64__)
static inline uint64_t rt_get_fpcr(void) { uint64_t v; __asm__ volatile("mrs %0, fpcr" : "=r"(v)); return v; }
static inline uint64_t rt_get_fpsr(void) { uint64_t v; __asm__ volatile("mrs %0, fpsr" : "=r"(v)); return v; }
static inline void rt_set_fpcr(uint64_t v) { __asm__ volatile("msr fpcr, %0" :: "r"(v)); }
static inline void rt_set_fpsr(uint64_t v) { __asm__ volatile("msr fpsr, %0" :: "r"(v)); }
#else
// Host FP state is not the guest's here: code lifted with --simd=c keeps FPCR/FPSR in Cpu (c->fpcr, zeroed
// with the Cpu), so the host-side reset in thread start-up has nothing to do.
static inline uint64_t rt_get_fpcr(void) { return 0; }
static inline uint64_t rt_get_fpsr(void) { return 0; }
static inline void rt_set_fpcr(uint64_t v) { (void)v; }
static inline void rt_set_fpsr(uint64_t v) { (void)v; }
#endif

// ---- memory: guest addresses are host addresses
typedef uint16_t __attribute__((aligned(1), may_alias)) u16u;
typedef uint32_t __attribute__((aligned(1), may_alias)) u32u;
typedef uint64_t __attribute__((aligned(1), may_alias)) u64u;
typedef uint8_t __attribute__((may_alias)) u8a;

// Function-entry profiler (runtime/fnprof.c, SAVAGEWIND_FNPROF): one predictable branch per call when off.
extern volatile int g_fnprof;
void fnprof_hit(uint64_t addr, uint64_t lr, uint64_t x0);
#define FN_ENTER(a) do { if (__builtin_expect(g_fnprof, 0)) fnprof_hit(a, c->x[30], c->x[0]); } while (0)
#define FN_ENTER_N(a) do { if (__builtin_expect(g_fnprof, 0)) fnprof_hit(a, cg->x[30], cg->x[0]); } while (0)  // (native)

// volatile: every guest load and store happens, in order, as on the console's CPU. Guest memory is shared with the
// game's other threads: with plain accesses the compiler could keep a spinlock's word in a register for good (BOTW's
// 0x7000f9449c spins on a plain LDR until the lock reads free: at -O2 it read it once and hung) or sink a store out of
// a loop.
// Through types of their own (GMem*), with strict aliasing (GENFLAGS): type-based alias analysis then knows that a
// guest load or store can't touch the Cpu struct. The addresses come from integers, so restrict on the Cpu couldn't
// show it, and every guest store made the compiler load the guest registers it held from the Cpu again (a function's
// prologue reloaded sp between each pair of registers it saved): ~3-5% of the game code's instructions.
typedef struct { uint8_t m; } __attribute__((packed)) GMem8;
typedef struct { uint16_t m; } __attribute__((packed)) GMem16;
typedef struct { uint32_t m; } __attribute__((packed)) GMem32;
typedef struct { uint64_t m; } __attribute__((packed)) GMem64;
// -DGUEST_MEM_PLAIN (BOTW-NATIVE-CODEGEN.md change 4): plain accesses, which the compiler may merge, forward and keep in
// registers, and a compiler barrier where the generated code branches back (tools/gen.py: every loop, a spin loop's
// too, reads memory again each time round). ARM reorders plain accesses itself, so guest code that is correct on the
// console can't rely on more; its atomics (LDXR/STXR, LDAR/STLR) and barriers (DMB) stay what they are.
#ifdef GUEST_MEM_PLAIN
#define GMEM_V
#define GUEST_LOOP_BARRIER() __asm__ volatile("" ::: "memory")
#else
#define GMEM_V volatile
#define GUEST_LOOP_BARRIER() ((void)0)
#endif
#define LD8(a)  (((const GMEM_V GMem8 *)(uintptr_t)(a))->m)
#define LD16(a) (((const GMEM_V GMem16 *)(uintptr_t)(a))->m)
#define LD32(a) (((const GMEM_V GMem32 *)(uintptr_t)(a))->m)
#define LD64(a) (((const GMEM_V GMem64 *)(uintptr_t)(a))->m)
#define ST8(a, v)  (((GMEM_V GMem8 *)(uintptr_t)(a))->m = (uint8_t)(v))
#define ST16(a, v) (((GMEM_V GMem16 *)(uintptr_t)(a))->m = (uint16_t)(v))
#define ST32(a, v) (((GMEM_V GMem32 *)(uintptr_t)(a))->m = (uint32_t)(v))
#define ST64(a, v) (((GMEM_V GMem64 *)(uintptr_t)(a))->m = (uint64_t)(v))

// ---- exclusive monitor (emulated with compare-and-swap)
#define EXCL(N, T)                                                                         \
    static inline uint64_t ldx##N(Cpu *c, uint64_t a) {                                    \
        T v = __atomic_load_n((T *)(uintptr_t)a, __ATOMIC_ACQUIRE);                        \
        c->excl_addr = a;                                                                  \
        c->excl_val = v;                                                                   \
        return v;                                                                          \
    }                                                                                      \
    static inline uint32_t stx##N(Cpu *c, uint64_t a, uint64_t v) {                        \
        if (c->excl_addr != a) return 1;                                                   \
        c->excl_addr = ~0ull;                                                              \
        T expected = (T)c->excl_val;                                                       \
        return !__atomic_compare_exchange_n((T *)(uintptr_t)a, &expected, (T)v, 0,         \
                                            __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);           \
    }                                                                                      \
    static inline uint64_t lda##N(uint64_t a) {                                            \
        return __atomic_load_n((T *)(uintptr_t)a, __ATOMIC_ACQUIRE);                       \
    }                                                                                      \
    static inline void stl##N(uint64_t a, uint64_t v) {                                    \
        __atomic_store_n((T *)(uintptr_t)a, (T)v, __ATOMIC_RELEASE);                       \
    }
EXCL(8, uint8_t)
EXCL(16, uint16_t)
EXCL(32, uint32_t)
EXCL(64, uint64_t)

// ---- flags (packed NZCV in bits 31..28)
#define FLAG_N(c) (((c)->nzcv >> 31) & 1)
#define FLAG_Z(c) (((c)->nzcv >> 30) & 1)
#define FLAG_C(c) (((c)->nzcv >> 29) & 1)
#define FLAG_V(c) (((c)->nzcv >> 28) & 1)
#define COND_EQ(c) (FLAG_Z(c))
#define COND_NE(c) (!FLAG_Z(c))
#define COND_CS(c) (FLAG_C(c))
#define COND_CC(c) (!FLAG_C(c))
#define COND_MI(c) (FLAG_N(c))
#define COND_PL(c) (!FLAG_N(c))
#define COND_VS(c) (FLAG_V(c))
#define COND_VC(c) (!FLAG_V(c))
#define COND_HI(c) (FLAG_C(c) && !FLAG_Z(c))
#define COND_LS(c) (!(FLAG_C(c) && !FLAG_Z(c)))
#define COND_GE(c) (FLAG_N(c) == FLAG_V(c))
#define COND_LT(c) (FLAG_N(c) != FLAG_V(c))
#define COND_GT(c) (!FLAG_Z(c) && FLAG_N(c) == FLAG_V(c))
#define COND_LE(c) (!(!FLAG_Z(c) && FLAG_N(c) == FLAG_V(c)))

#define MKFLAGS(n, z, cy, v) \
    (((uint64_t)(n) << 31) | ((uint64_t)(z) << 30) | ((uint64_t)(cy) << 29) | ((uint64_t)(v) << 28))

// The flag-setting ops, for the Cpu's flags (adds64(c, ...)) and for a function's own (nadds64(&fl, ...): native
// functions keep the flags in a local NFlags, tools/gen.py, BOTW-NATIVE-CODEGEN.md). COND_*/FLAG_* take either.
typedef struct { uint64_t nzcv; } NFlags;
#define FLAG_OPS(P, T)                                                                            \
    static inline uint64_t P##adds64(T *c, uint64_t a, uint64_t b) {                              \
        uint64_t r = a + b;                                                                       \
        c->nzcv = MKFLAGS(r >> 63, r == 0, r < a, ((a ^ r) & (b ^ r)) >> 63);                     \
        return r;                                                                                 \
    }                                                                                             \
    static inline uint64_t P##subs64(T *c, uint64_t a, uint64_t b) {                              \
        uint64_t r = a - b;                                                                       \
        c->nzcv = MKFLAGS(r >> 63, r == 0, a >= b, ((a ^ b) & (a ^ r)) >> 63);                    \
        return r;                                                                                 \
    }                                                                                             \
    static inline uint32_t P##adds32(T *c, uint32_t a, uint32_t b) {                              \
        uint32_t r = a + b;                                                                       \
        c->nzcv = MKFLAGS(r >> 31, r == 0, r < a, ((a ^ r) & (b ^ r)) >> 31);                     \
        return r;                                                                                 \
    }                                                                                             \
    static inline uint32_t P##subs32(T *c, uint32_t a, uint32_t b) {                              \
        uint32_t r = a - b;                                                                       \
        c->nzcv = MKFLAGS(r >> 31, r == 0, a >= b, ((a ^ b) & (a ^ r)) >> 31);                    \
        return r;                                                                                 \
    }                                                                                             \
    static inline uint64_t P##logic64(T *c, uint64_t r) { c->nzcv = MKFLAGS(r >> 63, r == 0, 0, 0); return r; } \
    static inline uint32_t P##logic32(T *c, uint32_t r) { c->nzcv = MKFLAGS(r >> 31, r == 0, 0, 0); return r; }
FLAG_OPS(, Cpu)
FLAG_OPS(n, NFlags)

#define ADC_IMPL(N, P, T)                                                                  \
    static inline uint##N##_t P##adc##N(T *c, uint##N##_t a, uint##N##_t b) {              \
        return a + b + (uint##N##_t)FLAG_C(c);                                             \
    }                                                                                      \
    static inline uint##N##_t P##sbc##N(T *c, uint##N##_t a, uint##N##_t b) {              \
        return a - b - 1 + (uint##N##_t)FLAG_C(c);                                         \
    }                                                                                      \
    static inline uint##N##_t P##adcs##N(T *c, uint##N##_t a, uint##N##_t b) {             \
        uint##N##_t cin = FLAG_C(c), r = a + b + cin;                                      \
        int carry = cin ? r <= a : r < a;                                                  \
        c->nzcv = MKFLAGS(r >> (N - 1), r == 0, carry, ((a ^ r) & (b ^ r)) >> (N - 1));    \
        return r;                                                                          \
    }                                                                                      \
    static inline uint##N##_t P##sbcs##N(T *c, uint##N##_t a, uint##N##_t b) {             \
        return P##adcs##N(c, a, (uint##N##_t)~b);                                          \
    }
ADC_IMPL(64, , Cpu)
ADC_IMPL(32, , Cpu)
ADC_IMPL(64, n, NFlags)
ADC_IMPL(32, n, NFlags)

// ---- bit ops
#define ROR64(v, r) ((uint64_t)(v) >> ((r) & 63) | (uint64_t)(v) << ((64 - ((r) & 63)) & 63))
#define ROR32(v, r) ((uint32_t)(v) >> ((r) & 31) | (uint32_t)(v) << ((32 - ((r) & 31)) & 31))
static inline uint64_t clz64(uint64_t v) { return v ? __builtin_clzll(v) : 64; }
static inline uint32_t clz32(uint32_t v) { return v ? __builtin_clz(v) : 32; }
static inline uint64_t cls64(uint64_t v) { return clz64(v ^ (uint64_t)((int64_t)v >> 1)) - 1; }
static inline uint32_t cls32(uint32_t v) { return clz32(v ^ (uint32_t)((int32_t)v >> 1)) - 1; }
static inline uint64_t rbit64(uint64_t v) { return __builtin_bitreverse64(v); }
static inline uint32_t rbit32(uint32_t v) { return __builtin_bitreverse32(v); }
static inline uint32_t rev16_32(uint32_t v) { return ((v & 0x00ff00ffu) << 8) | ((v >> 8) & 0x00ff00ffu); }
static inline uint64_t rev16_64(uint64_t v) {
    return ((v & 0x00ff00ff00ff00ffull) << 8) | ((v >> 8) & 0x00ff00ff00ff00ffull);
}
static inline uint64_t rev32_64(uint64_t v) {
    return ((uint64_t)__builtin_bswap32((uint32_t)v)) | ((uint64_t)__builtin_bswap32((uint32_t)(v >> 32)) << 32);
}
static inline uint64_t udiv64(uint64_t a, uint64_t b) { return b ? a / b : 0; }
static inline uint32_t udiv32(uint32_t a, uint32_t b) { return b ? a / b : 0; }
static inline uint64_t sdiv64(uint64_t a, uint64_t b) {
    if (!b) return 0;
    if ((int64_t)a == INT64_MIN && (int64_t)b == -1) return a;
    return (uint64_t)((int64_t)a / (int64_t)b);
}
static inline uint32_t sdiv32(uint32_t a, uint32_t b) {
    if (!b) return 0;
    if ((int32_t)a == INT32_MIN && (int32_t)b == -1) return a;
    return (uint32_t)((int32_t)a / (int32_t)b);
}

// Portable FP/SIMD helpers for code lifted with --simd=c (default off AArch64; gen.py includes simd.h itself
// in that mode, so AArch64 hosts can build it too).
#if !defined(__cplusplus) && !defined(__aarch64__)
#include "simd.h"
#endif
