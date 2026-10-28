#include "../../perf_survey.h"
// Command recording on a thread of its own (vkq.h). The ring holds entries of a 16-byte header (the replay function,
// the entry's size) and the call's arguments, the arrays its pointers point to right after them; a pointer inside is
// rebuilt at replay from where its array sits in the entry. One producer (the submission thread, while it translates
// a list) and one consumer (the recording thread).
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <immintrin.h>
#ifdef _WIN32
#include <windows.h>
#endif

#define VKD_NO_REDIRECT
#include "vkd.h"
#include "vkq.h"
#include "timing.h"
#include "a1d.h"
#include "frame_trace.h"
#include "../../rt.h"

__thread bool vkq_on;
void (*vkq_work_hook)(VkCommandBuffer cb);
#define WORK(cb) do { if (vkq_work_hook) vkq_work_hook(cb); } while (0)
bool vkq_address_wait;
bool vkq_native_multi_draw;
void vk_native_batch_replay(void *);
void vk_native_batch_flush(void);
static void R_NativeFlush(void *unused) { (void)unused; vk_native_batch_flush(); }
#ifdef _WIN32
static BOOL (WINAPI *wait_address_fn)(volatile VOID *, PVOID, SIZE_T, DWORD);
static VOID (WINAPI *wake_address_one_fn)(PVOID), (WINAPI *wake_address_all_fn)(PVOID);
#endif

void vkq_wait_address(volatile void *address, uint64_t expected, size_t bytes) {
#ifdef _WIN32
    // Caller always rechecks its acquire-loaded condition, including on a
    // spurious wake. WaitOnAddress compares before sleeping, so early wakes are safe.
    wait_address_fn(address, &expected, bytes, INFINITE);
#else
    (void)address; (void)expected; (void)bytes;
#endif
}

void vkq_wake_address(void *address, bool all) {
#ifdef _WIN32
    if (all) wake_address_all_fn(address); else wake_address_one_fn(address);
#else
    (void)address; (void)all;
#endif
}

typedef void (*ReplayFn)(void *args);
typedef struct { ReplayFn fn; uint32_t size, pad; } Hdr;  // size: the whole entry, a multiple of 16
static void R_CmdDraw(void *), R_CmdDrawIndexed(void *), R_CmdBindIndexBuffer(void *);
static void R_CmdBindVertexBuffers(void *), R_CmdBindDescriptorSets(void *);
static __thread bool batch_probe;
static __thread uint64_t batch_probe_token;
static __thread ReplayFn batch_probe_descriptor_bind;
uint64_t vkq_batch_probe_token(void (*descriptor_bind)(void *)) {
    batch_probe = true;
    batch_probe_descriptor_bind = descriptor_bind;
    return batch_probe_token;
}
#define RING_SIZE (64u << 20)

static uint8_t *ring;
// bytes written / replayed since the start (never wrap). Each on a cache line of its own: the two threads write them
// ~20 times a draw, and on one line every write took the line from the other core.
static _Alignas(64) _Atomic uint64_t wpos;
static _Alignas(64) _Atomic uint64_t rpos;
static _Alignas(64) uint64_t wlocal;  // (producer: written, not yet published)
static uint64_t rcache;               // (producer: rpos as last read; read again only when the ring looks full)
static pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t cv_work = PTHREAD_COND_INITIALIZER, cv_idle = PTHREAD_COND_INITIALIZER;
static _Alignas(64) _Atomic int sleeping, waiting;
static int state;  // 0 not started, 1 on, -1 off (SAVAGEWIND_VK_NO_DEFER, or no thread)
static _Atomic int failed;  // a deferred submission failed: the producer reports it at the next drain
static _Atomic uint64_t trace_ring_ns, trace_drain_ns;
_Atomic uint64_t vkq_idle_ns, vkq_submit_ns, vkq_cycles;  // (SAVAGEWIND_VK_PAR_PROBE, exec.c) the recording thread waiting for work; in vkQueueSubmit2
bool vkq_probe;
uint64_t renderer_present_count(void);

void vkq_trace_frame(uint64_t frame) {
    static uint64_t previous[2];
    uint64_t current[2] = {atomic_load_explicit(&trace_ring_ns, memory_order_relaxed),
                           atomic_load_explicit(&trace_drain_ns, memory_order_relaxed)};
    TRACE_LOG("[vkq-work] %llu recorder_ring_wait_ns %llu recorder_drain_wait_ns %llu\n",
              (unsigned long long)frame, (unsigned long long)(current[0] - previous[0]),
              (unsigned long long)(current[1] - previous[1]));
    memcpy(previous, current, sizeof previous);
}

void kernel_sample_host_thread(void);
static void *recorder(void *arg) {
    (void)arg;
    kernel_sample_host_thread();  // (SAVAGEWIND_SAMPLE profiles it as a host thread)
    uint64_t a1d_busy=0;unsigned a1d_ops=0;
    for (;;) {
        uint64_t r = atomic_load_explicit(&rpos, memory_order_relaxed);
        uint64_t w = atomic_load_explicit(&wpos, memory_order_acquire);
        if (r == w) {
            uint64_t idle0 = vkq_probe ? plat_now_ns() : 0;
            if (idle0) atomic_store_explicit(&vkq_cycles, perf_thread_cycles(), memory_order_relaxed);
            if(a1d_busy){a1d_local[A1D_RECORDER]+=plat_now_ns()-a1d_busy;a1d_busy=0;a1d_publish();}
            if (vkq_address_wait) {
                if (atomic_load(&waiting)) vkq_wake_address(&rpos, false);
                for (int i = 0; i < 64 && atomic_load_explicit(&wpos, memory_order_acquire) == r; i++) _mm_pause();
                if (atomic_load_explicit(&wpos, memory_order_acquire) == r) vkq_wait_address(&wpos, r, sizeof wpos);
                if (idle0) atomic_fetch_add_explicit(&vkq_idle_ns, plat_now_ns() - idle0, memory_order_relaxed);
                continue;
            }
            if (atomic_load(&waiting)) { pthread_mutex_lock(&m); pthread_cond_broadcast(&cv_idle); pthread_mutex_unlock(&m); }
            bool got = false;
            // (a short spin: the next command is usually microseconds away; PERFX_RECSPIN: 200 pauses in the on blocks)
            int spin = perfx_trial(PERFX_RECSPIN) && perfx(PERFX_RECSPIN) ? 200 : 4000;
            for (int i = 0; i < spin && !got; i++) {
                _mm_pause();
                got = atomic_load_explicit(&wpos, memory_order_acquire) != r;
            }
            if (!got) {
                pthread_mutex_lock(&m);
                atomic_store(&sleeping, 1);
                while (atomic_load_explicit(&wpos, memory_order_acquire) == r) pthread_cond_wait(&cv_work, &m);
                atomic_store(&sleeping, 0);
                pthread_mutex_unlock(&m);
            }
            if (idle0) atomic_fetch_add_explicit(&vkq_idle_ns, plat_now_ns() - idle0, memory_order_relaxed);
            continue;
        }
        if(a1d_enabled&&!a1d_busy)a1d_busy=plat_now_ns();
        Hdr *h = (Hdr *)(ring + (r % RING_SIZE));
        if (h->fn) {
            perf_survey_delay(SURVEY_RECORDER, 0);
            if(vkq_native_multi_draw&&h->fn!=vk_native_batch_replay)vk_native_batch_flush();
            h->fn(h + 1);
        }  // (fn NULL: padding to the end of the ring)
        atomic_store_explicit(&rpos, r + h->size, memory_order_release);
        if(a1d_busy&&!(++a1d_ops&63)){a1d_local[A1D_RECORDER]+=plat_now_ns()-a1d_busy;a1d_busy=0;a1d_publish();}
    }
    return NULL;
}

static bool start(void) {
    if (state) return state > 0;
    state = -1;
    if (getenv("SAVAGEWIND_VK_NO_DEFER")) { LOG("[vk] command recording on the submission thread (SAVAGEWIND_VK_NO_DEFER)\n"); return false; }
#ifdef _WIN32
    const char *native = getenv("SAVAGEWIND_VK_ADDRESS_WAIT");
    if (native && atoi(native)) {
        HMODULE kernel = GetModuleHandleA("kernelbase.dll");
        wait_address_fn = (void *)GetProcAddress(kernel, "WaitOnAddress");
        wake_address_one_fn = (void *)GetProcAddress(kernel, "WakeByAddressSingle");
        wake_address_all_fn = (void *)GetProcAddress(kernel, "WakeByAddressAll");
        vkq_address_wait = wait_address_fn && wake_address_one_fn && wake_address_all_fn;
    }
#endif
    ring = malloc(RING_SIZE);
    pthread_t t;
    if (!ring || pthread_create(&t, NULL, recorder, NULL)) { LOG("[vk] no recording thread: commands recorded in place\n"); return false; }
    pthread_detach(t);
    state = 1;
    LOG("[vk] commands recorded on a thread of their own while lists are translated\n");
    if (vkq_address_wait) LOG("[vk] native address waits for recorder and descriptor jobs\n");
    return true;
}

static void publish(void) {
    atomic_store_explicit(&wpos, wlocal, memory_order_release);
    if (vkq_address_wait) { vkq_wake_address(&wpos, false); return; }
    if (atomic_load(&sleeping)) { pthread_mutex_lock(&m); pthread_cond_signal(&cv_work); pthread_mutex_unlock(&m); }
}

// An entry of `args` bytes for fn, its argument block returned (16-aligned). The entries before it (filled by now)
// are published in batches of 2 KB (about two draws): publishing each took a cache line from the recorder every call.
// Submissions and drains publish at once.
static void *push(ReplayFn fn, size_t args) {
    if (batch_probe && fn != R_CmdDraw && fn != R_CmdDrawIndexed &&
        fn != R_CmdBindIndexBuffer && fn != R_CmdBindVertexBuffers &&
        fn != R_CmdBindDescriptorSets && fn != batch_probe_descriptor_bind)
        batch_probe_token++;
    if (wlocal - atomic_load_explicit(&wpos, memory_order_relaxed) >= 2048) publish();
    uint32_t size = (uint32_t)((sizeof(Hdr) + args + 15) & ~(size_t)15);
    uint64_t off = wlocal % RING_SIZE;
    uint32_t pad = off + size > RING_SIZE ? (uint32_t)(RING_SIZE - off) : 0;  // (an entry never wraps)
    uint64_t trace_wait = 0;
    while (wlocal + pad + size - rcache > RING_SIZE) {
        rcache = atomic_load_explicit(&rpos, memory_order_acquire);
        if (wlocal + pad + size - rcache <= RING_SIZE) break;
        if (vk_frame_trace_on && !trace_wait) trace_wait = plat_now_ns();
        publish();
        _mm_pause();
    }
    if (trace_wait) {
        uint64_t end = plat_now_ns();
        atomic_fetch_add_explicit(&trace_ring_ns, end - trace_wait, memory_order_relaxed);
        if (end - trace_wait >= 25000) vk_frame_trace_event("Vulkan command ring full", 1,
            renderer_present_count(), 0, trace_wait, end, 0, 0);
    }
    if (pad) {
        Hdr *p = (Hdr *)(ring + off);
        p->fn = NULL;
        p->size = pad;
        wlocal += pad;
    }
    Hdr *h = (Hdr *)(ring + (wlocal % RING_SIZE));
    h->fn = fn;
    h->size = size;
    wlocal += size;
    return h + 1;
}

void vkq_drain(void) {
    if (state <= 0) return;
    if(vkq_native_multi_draw)push(R_NativeFlush,0);
    publish();
    uint64_t trace_wait = vk_frame_trace_on && atomic_load_explicit(&rpos, memory_order_acquire) != wlocal ? plat_now_ns() : 0;
    if (vkq_address_wait) {
        atomic_store(&waiting, 1);
        uint64_t r;
        while ((r = atomic_load_explicit(&rpos, memory_order_acquire)) != wlocal)
            vkq_wait_address(&rpos, r, sizeof rpos);
        atomic_store(&waiting, 0);
    } else if (atomic_load_explicit(&rpos, memory_order_acquire) != wlocal) {
        pthread_mutex_lock(&m);
        atomic_store(&waiting, 1);
        while (atomic_load_explicit(&rpos, memory_order_acquire) != wlocal) {
            if (atomic_load(&sleeping)) pthread_cond_signal(&cv_work);
            struct timespec ts;  // (a timed wait: the recorder broadcasts when idle, this only guards a missed one)
            clock_gettime(CLOCK_REALTIME, &ts);
            ts.tv_nsec += 2000000;
            if (ts.tv_nsec >= 1000000000) { ts.tv_sec++; ts.tv_nsec -= 1000000000; }
            pthread_cond_timedwait(&cv_idle, &m, &ts);
        }
        atomic_store(&waiting, 0);
        pthread_mutex_unlock(&m);
    }
    if (trace_wait) {
        uint64_t end = plat_now_ns();
        atomic_fetch_add_explicit(&trace_drain_ns, end - trace_wait, memory_order_relaxed);
        if (end - trace_wait >= 25000) vk_frame_trace_event("Drain Vulkan command recorder", 1,
            renderer_present_count(), 0, trace_wait, end, 0, 0);
    }
    if (atomic_exchange(&failed, 0)) FATAL("deferred vkQueueSubmit2 failed (see the log above)\n");
}

void vkq_defer_begin(void) {
    if (start()) vkq_on = true;
}

void vkq_defer_end(void) {
    if (!vkq_on) return;
    vkq_drain();
    vkq_on = false;
}

// Deferring ends without waiting for the recording thread, which goes on with what was deferred; whoever calls Vulkan
// directly next (any thread, with exec.c's gpu_lock held) waits for it first: vkq_catch_up.
static _Atomic uint64_t detached;  // (the ring position the producer published as it stopped)
void vkq_defer_detach(void) {
    if (!vkq_on) return;
    if (vkq_native_multi_draw) push(R_NativeFlush, 0);
    publish();
    atomic_store_explicit(&detached, wlocal, memory_order_release);
    vkq_on = false;
}
void vkq_catch_up(void) {
    if (state <= 0 || vkq_on) return;  // (deferring: the producer's own calls are in order anyway)
    uint64_t d = atomic_load_explicit(&detached, memory_order_acquire);
    if (atomic_load_explicit(&rpos, memory_order_acquire) >= d) return;
    uint64_t trace_wait = vk_frame_trace_on ? plat_now_ns() : 0;
    if (vkq_address_wait) {
        atomic_store(&waiting, 1);
        uint64_t r;
        while ((r = atomic_load_explicit(&rpos, memory_order_acquire)) < d) vkq_wait_address(&rpos, r, sizeof rpos);
        atomic_store(&waiting, 0);
    } else {
        pthread_mutex_lock(&m);
        atomic_store(&waiting, 1);
        while (atomic_load_explicit(&rpos, memory_order_acquire) < d) {
            if (atomic_load(&sleeping)) pthread_cond_signal(&cv_work);
            struct timespec ts;  // (as vkq_drain)
            clock_gettime(CLOCK_REALTIME, &ts);
            ts.tv_nsec += 2000000;
            if (ts.tv_nsec >= 1000000000) { ts.tv_sec++; ts.tv_nsec -= 1000000000; }
            pthread_cond_timedwait(&cv_idle, &m, &ts);
        }
        atomic_store(&waiting, 0);
        pthread_mutex_unlock(&m);
    }
    if (trace_wait) {
        uint64_t end = plat_now_ns();
        atomic_fetch_add_explicit(&trace_drain_ns, end - trace_wait, memory_order_relaxed);
    }
    if (atomic_exchange(&failed, 0)) FATAL("deferred vkQueueSubmit2 failed (see the log above)\n");
}

// ---- an entry of the caller's own: fn(arg) on the recording thread, in order with the commands around it (exec.c: a
// draw's descriptor sets, which a worker thread builds meanwhile). size bytes of arg are copied into the entry.
bool vkq_custom(void (*fn)(void *), const void *arg, size_t size) {
    if (!vkq_on) return false;
    void *a = push((ReplayFn)fn, size);
    memcpy(a, arg, size);
    return true;
}

// ---- argument blocks: what follows a struct in its entry
#define AFTER(e) ((uint8_t *)((e) + 1))
#define ARR(T, n) (((size_t)(n) * sizeof(T) + 15) & ~(size_t)15)

typedef struct {
    VkCommandBuffer cb;
    VkPipelineLayout layout;
    VkDescriptorSet sets[2];
    void (*bind_job)(VkCommandBuffer, VkPipelineLayout, void *);
    void *job;
    VkBuffer index;
    VkDeviceSize index_offset;
    VkQueryPool query_pool;
    uint32_t ndyn, vertex_mask, flags, count, instances, first, base_instance;
    int32_t base_vertex;
    VkIndexType index_type;
    uint32_t query;
    VkQueryControlFlags query_flags;
} DrawPacket;

static void R_DrawPacket(void *p) {
    DrawPacket *a = p;
    unsigned nv = (unsigned)__builtin_popcount(a->vertex_mask);
    const uint32_t *dyn = (const uint32_t *)AFTER(a);
    const VkBuffer *buffers = (const VkBuffer *)(AFTER(a) + ARR(uint32_t, a->ndyn));
    const VkDeviceSize *offsets = (const VkDeviceSize *)((const uint8_t *)buffers + ARR(VkBuffer, nv));
    if (a->bind_job) a->bind_job(a->cb, a->layout, a->job);
    else if (a->flags & 1) vkd_CmdBindDescriptorSets(a->cb, VK_PIPELINE_BIND_POINT_GRAPHICS, a->layout, 0, 2, a->sets, a->ndyn, dyn);
    if (a->flags & 2) vkd_CmdBindIndexBuffer(a->cb, a->index, a->index_offset, a->index_type);
    // Adjacent changed streams bind together; gaps retain their existing state.
    unsigned pos = 0;
    for (unsigned s = 0; s < 17;) {
        if (!(a->vertex_mask & (1u << s))) { s++; continue; }
        unsigned first = s++;
        while (s < 17 && (a->vertex_mask & (1u << s))) s++;
        unsigned n = s - first;
        vkd_CmdBindVertexBuffers(a->cb, first, n, buffers + pos, offsets + pos);
        pos += n;
    }
    if (a->query_pool) vkd_CmdBeginQuery(a->cb, a->query_pool, a->query, a->query_flags);
    if (a->flags & 4) vkd_CmdDrawIndexed(a->cb, a->count, a->instances, a->first, a->base_vertex, a->base_instance);
    else vkd_CmdDraw(a->cb, a->count, a->instances, a->first, a->base_instance);
    if (a->query_pool) vkd_CmdEndQuery(a->cb, a->query_pool, a->query);
}

void vkq_draw_packet(const VkqDraw *d) {
    uint32_t mask = d->vertex_mask & ((1u << 17) - 1);
    unsigned nv = (unsigned)__builtin_popcount(mask);
    unsigned ndyn = d->bind_sets && !d->bind_job ? d->ndyn : 0;
    if (ndyn > 32) FATAL("native draw packet: too many dynamic offsets\n");
    size_t size = sizeof(DrawPacket) + ARR(uint32_t, ndyn) + ARR(VkBuffer, nv) + ARR(VkDeviceSize, nv);
    _Alignas(16) uint8_t local[1024];
    _Static_assert(sizeof(DrawPacket) + 128 + 2 * 144 <= sizeof local, "draw packet scratch");
    DrawPacket *a = vkq_on ? push(R_DrawPacket, size) : (DrawPacket *)local;
    *a = (DrawPacket){.cb = d->cb, .layout = d->layout, .sets = {d->sets[0], d->sets[1]},
        .bind_job = d->bind_job, .job = d->job, .index = d->index, .index_offset = d->index_offset,
        .query_pool = d->query_pool, .ndyn = ndyn, .vertex_mask = mask,
        .flags = (d->bind_sets ? 1u : 0) | (d->bind_index ? 2u : 0) | (d->indexed ? 4u : 0),
        .count = d->count, .instances = d->instances, .first = d->first, .base_instance = d->base_instance,
        .base_vertex = d->base_vertex, .index_type = d->index_type, .query = d->query, .query_flags = d->query_flags};
    memcpy(AFTER(a), d->dyn, ndyn * sizeof(uint32_t));
    VkBuffer *buffers = (VkBuffer *)(AFTER(a) + ARR(uint32_t, ndyn));
    VkDeviceSize *offsets = (VkDeviceSize *)((uint8_t *)buffers + ARR(VkBuffer, nv));
    unsigned pos = 0;
    for (unsigned s = 0; s < 17; s++) if (mask & (1u << s)) {
        buffers[pos] = d->vertex[s]; offsets[pos++] = d->offsets[s];
    }
    if (!vkq_on) R_DrawPacket(a);
}

// ---- scalar calls
#define SCALAR0(name)                                                                                                 \
    typedef struct { VkCommandBuffer cb; } A_##name;                                                                  \
    static void R_##name(void *p) { A_##name *a = p; vkd_##name(a->cb); }                                            \
    VKAPI_ATTR void VKAPI_CALL vkq_##name(VkCommandBuffer cb) {                                                       \
        if (!vkq_on) { vkd_##name(cb); return; }                                                                      \
        A_##name *a = push(R_##name, sizeof *a);                                                                      \
        a->cb = cb;                                                                                                   \
    }
#define SCALAR1(name, T1)                                                                                             \
    typedef struct { VkCommandBuffer cb; T1 a1; } A_##name;                                                           \
    static void R_##name(void *p) { A_##name *a = p; vkd_##name(a->cb, a->a1); }                                     \
    VKAPI_ATTR void VKAPI_CALL vkq_##name(VkCommandBuffer cb, T1 a1) {                                                \
        if (!vkq_on) { vkd_##name(cb, a1); return; }                                                                  \
        A_##name *a = push(R_##name, sizeof *a);                                                                      \
        a->cb = cb; a->a1 = a1;                                                                                       \
    }
#define SCALAR2(name, T1, T2)                                                                                         \
    typedef struct { VkCommandBuffer cb; T1 a1; T2 a2; } A_##name;                                                    \
    static void R_##name(void *p) { A_##name *a = p; vkd_##name(a->cb, a->a1, a->a2); }                              \
    VKAPI_ATTR void VKAPI_CALL vkq_##name(VkCommandBuffer cb, T1 a1, T2 a2) {                                         \
        if (!vkq_on) { vkd_##name(cb, a1, a2); return; }                                                              \
        A_##name *a = push(R_##name, sizeof *a);                                                                      \
        a->cb = cb; a->a1 = a1; a->a2 = a2;                                                                           \
    }
#define SCALAR3(name, T1, T2, T3)                                                                                     \
    typedef struct { VkCommandBuffer cb; T1 a1; T2 a2; T3 a3; } A_##name;                                             \
    static void R_##name(void *p) { A_##name *a = p; vkd_##name(a->cb, a->a1, a->a2, a->a3); }                       \
    VKAPI_ATTR void VKAPI_CALL vkq_##name(VkCommandBuffer cb, T1 a1, T2 a2, T3 a3) {                                  \
        if (!vkq_on) { vkd_##name(cb, a1, a2, a3); return; }                                                          \
        A_##name *a = push(R_##name, sizeof *a);                                                                      \
        a->cb = cb; a->a1 = a1; a->a2 = a2; a->a3 = a3;                                                               \
    }
#define SCALAR4(name, T1, T2, T3, T4)                                                                                 \
    typedef struct { VkCommandBuffer cb; T1 a1; T2 a2; T3 a3; T4 a4; } A_##name;                                      \
    static void R_##name(void *p) { A_##name *a = p; vkd_##name(a->cb, a->a1, a->a2, a->a3, a->a4); }                \
    VKAPI_ATTR void VKAPI_CALL vkq_##name(VkCommandBuffer cb, T1 a1, T2 a2, T3 a3, T4 a4) {                           \
        if (!vkq_on) { vkd_##name(cb, a1, a2, a3, a4); return; }                                                      \
        A_##name *a = push(R_##name, sizeof *a);                                                                      \
        a->cb = cb; a->a1 = a1; a->a2 = a2; a->a3 = a3; a->a4 = a4;                                                   \
    }
#define SCALAR5(name, T1, T2, T3, T4, T5)                                                                             \
    typedef struct { VkCommandBuffer cb; T1 a1; T2 a2; T3 a3; T4 a4; T5 a5; } A_##name;                               \
    static void R_##name(void *p) { A_##name *a = p; vkd_##name(a->cb, a->a1, a->a2, a->a3, a->a4, a->a5); }         \
    VKAPI_ATTR void VKAPI_CALL vkq_##name(VkCommandBuffer cb, T1 a1, T2 a2, T3 a3, T4 a4, T5 a5) {                    \
        if (!vkq_on) { vkd_##name(cb, a1, a2, a3, a4, a5); return; }                                                  \
        A_##name *a = push(R_##name, sizeof *a);                                                                      \
        a->cb = cb; a->a1 = a1; a->a2 = a2; a->a3 = a3; a->a4 = a4; a->a5 = a5;                                       \
    }

SCALAR0(CmdEndRendering)
SCALAR1(CmdSetCullMode, VkCullModeFlags)
SCALAR1(CmdSetDepthBiasEnable, VkBool32)
SCALAR1(CmdSetDepthCompareOp, VkCompareOp)
SCALAR1(CmdSetDepthTestEnable, VkBool32)
SCALAR1(CmdSetDepthWriteEnable, VkBool32)
SCALAR1(CmdSetFrontFace, VkFrontFace)
SCALAR1(CmdSetLineWidth, float)
SCALAR1(CmdSetPrimitiveRestartEnable, VkBool32)
SCALAR1(CmdSetPrimitiveTopology, VkPrimitiveTopology)
SCALAR1(CmdSetStencilTestEnable, VkBool32)
SCALAR2(CmdBindPipeline, VkPipelineBindPoint, VkPipeline)
SCALAR2(CmdEndQuery, VkQueryPool, uint32_t)
SCALAR2(CmdSetStencilCompareMask, VkStencilFaceFlags, uint32_t)
SCALAR2(CmdSetStencilReference, VkStencilFaceFlags, uint32_t)
SCALAR2(CmdSetStencilWriteMask, VkStencilFaceFlags, uint32_t)
SCALAR3(CmdBeginQuery, VkQueryPool, uint32_t, VkQueryControlFlags)
SCALAR3(CmdBindIndexBuffer, VkBuffer, VkDeviceSize, VkIndexType)
SCALAR3(CmdResetQueryPool, VkQueryPool, uint32_t, uint32_t)
SCALAR3(CmdSetDepthBias, float, float, float)
SCALAR3(CmdWriteTimestamp2, VkPipelineStageFlags2, VkQueryPool, uint32_t)
SCALAR4(CmdDraw, uint32_t, uint32_t, uint32_t, uint32_t)
SCALAR5(CmdDrawIndexed, uint32_t, uint32_t, uint32_t, int32_t, uint32_t)
SCALAR5(CmdSetStencilOp, VkStencilFaceFlags, VkStencilOp, VkStencilOp, VkStencilOp, VkCompareOp)
typedef struct { VkCommandBuffer cb; uint32_t x, y, z; } A_CmdDispatch;
static void R_CmdDispatch(void *p) { A_CmdDispatch *a = p; vkd_CmdDispatch(a->cb, a->x, a->y, a->z); }
VKAPI_ATTR void VKAPI_CALL vkq_CmdDispatch(VkCommandBuffer cb, uint32_t x, uint32_t y, uint32_t z) {
    WORK(cb);
    if (!vkq_on) { vkd_CmdDispatch(cb, x, y, z); return; }
    A_CmdDispatch *a = push(R_CmdDispatch, sizeof *a);
    a->cb = cb; a->x = x; a->y = y; a->z = z;
}

// ---- calls with arrays
typedef struct { VkCommandBuffer cb; float c[4]; } A_CmdSetBlendConstants;
static void R_CmdSetBlendConstants(void *p) { A_CmdSetBlendConstants *a = p; vkd_CmdSetBlendConstants(a->cb, a->c); }
VKAPI_ATTR void VKAPI_CALL vkq_CmdSetBlendConstants(VkCommandBuffer cb, const float c[4]) {
    if (!vkq_on) { vkd_CmdSetBlendConstants(cb, c); return; }
    A_CmdSetBlendConstants *a = push(R_CmdSetBlendConstants, sizeof *a);
    a->cb = cb;
    memcpy(a->c, c, sizeof a->c);
}

#define FIRST_N_ARRAY(name, T)                                                                                        \
    typedef struct { VkCommandBuffer cb; uint32_t first, n; } A_##name;                                               \
    static void R_##name(void *p) { A_##name *a = p; vkd_##name(a->cb, a->first, a->n, (const T *)AFTER(a)); }       \
    VKAPI_ATTR void VKAPI_CALL vkq_##name(VkCommandBuffer cb, uint32_t first, uint32_t n, const T *v) {               \
        if (!vkq_on) { vkd_##name(cb, first, n, v); return; }                                                         \
        A_##name *a = push(R_##name, sizeof *a + ARR(T, n));                                                          \
        a->cb = cb; a->first = first; a->n = n;                                                                       \
        memcpy(AFTER(a), v, (size_t)n * sizeof(T));                                                                   \
    }
FIRST_N_ARRAY(CmdSetViewport, VkViewport)
FIRST_N_ARRAY(CmdSetScissor, VkRect2D)

typedef struct { VkCommandBuffer cb; uint32_t first, n; } A_CmdBindVertexBuffers;
static void R_CmdBindVertexBuffers(void *p) {
    A_CmdBindVertexBuffers *a = p;
    const VkBuffer *b = (const VkBuffer *)AFTER(a);
    vkd_CmdBindVertexBuffers(a->cb, a->first, a->n, b, (const VkDeviceSize *)((const uint8_t *)b + ARR(VkBuffer, a->n)));
}
VKAPI_ATTR void VKAPI_CALL vkq_CmdBindVertexBuffers(VkCommandBuffer cb, uint32_t first, uint32_t n, const VkBuffer *b,
                                                    const VkDeviceSize *off) {
    if (!vkq_on) { vkd_CmdBindVertexBuffers(cb, first, n, b, off); return; }
    A_CmdBindVertexBuffers *a = push(R_CmdBindVertexBuffers, sizeof *a + ARR(VkBuffer, n) + ARR(VkDeviceSize, n));
    a->cb = cb; a->first = first; a->n = n;
    memcpy(AFTER(a), b, (size_t)n * sizeof *b);
    memcpy(AFTER(a) + ARR(VkBuffer, n), off, (size_t)n * sizeof *off);
}

typedef struct { VkCommandBuffer cb; VkPipelineBindPoint bp; VkPipelineLayout l; uint32_t first, n, ndyn; } A_CmdBindDescriptorSets;
static void R_CmdBindDescriptorSets(void *p) {
    A_CmdBindDescriptorSets *a = p;
    const VkDescriptorSet *s = (const VkDescriptorSet *)AFTER(a);
    vkd_CmdBindDescriptorSets(a->cb, a->bp, a->l, a->first, a->n, s, a->ndyn,
                              (const uint32_t *)((const uint8_t *)s + ARR(VkDescriptorSet, a->n)));
}
VKAPI_ATTR void VKAPI_CALL vkq_CmdBindDescriptorSets(VkCommandBuffer cb, VkPipelineBindPoint bp, VkPipelineLayout l,
                                                     uint32_t first, uint32_t n, const VkDescriptorSet *sets,
                                                     uint32_t ndyn, const uint32_t *dyn) {
    if (!vkq_on) { vkd_CmdBindDescriptorSets(cb, bp, l, first, n, sets, ndyn, dyn); return; }
    A_CmdBindDescriptorSets *a = push(R_CmdBindDescriptorSets, sizeof *a + ARR(VkDescriptorSet, n) + ARR(uint32_t, ndyn));
    a->cb = cb; a->bp = bp; a->l = l; a->first = first; a->n = n; a->ndyn = ndyn;
    memcpy(AFTER(a), sets, (size_t)n * sizeof *sets);
    if (ndyn) memcpy(AFTER(a) + ARR(VkDescriptorSet, n), dyn, (size_t)ndyn * sizeof *dyn);
}

typedef struct { VkCommandBuffer cb; VkPipelineLayout l; VkShaderStageFlags st; uint32_t off, size; } A_CmdPushConstants;
static void R_CmdPushConstants(void *p) {
    A_CmdPushConstants *a = p;
    vkd_CmdPushConstants(a->cb, a->l, a->st, a->off, a->size, AFTER(a));
}
VKAPI_ATTR void VKAPI_CALL vkq_CmdPushConstants(VkCommandBuffer cb, VkPipelineLayout l, VkShaderStageFlags st,
                                                uint32_t off, uint32_t size, const void *data) {
    if (!vkq_on) { vkd_CmdPushConstants(cb, l, st, off, size, data); return; }
    A_CmdPushConstants *a = push(R_CmdPushConstants, sizeof *a + ARR(uint8_t, size));
    a->cb = cb; a->l = l; a->st = st; a->off = off; a->size = size;
    memcpy(AFTER(a), data, size);
}

// copies: (src, src layout, dst, dst layout, regions) and the like
typedef struct { VkCommandBuffer cb; VkBuffer s, d; uint32_t n; } A_CmdCopyBuffer;
static void R_CmdCopyBuffer(void *p) { A_CmdCopyBuffer *a = p; vkd_CmdCopyBuffer(a->cb, a->s, a->d, a->n, (const VkBufferCopy *)AFTER(a)); }
VKAPI_ATTR void VKAPI_CALL vkq_CmdCopyBuffer(VkCommandBuffer cb, VkBuffer s, VkBuffer d, uint32_t n, const VkBufferCopy *r) {
    WORK(cb);
    if (!vkq_on) { vkd_CmdCopyBuffer(cb, s, d, n, r); return; }
    A_CmdCopyBuffer *a = push(R_CmdCopyBuffer, sizeof *a + ARR(VkBufferCopy, n));
    a->cb = cb; a->s = s; a->d = d; a->n = n;
    memcpy(AFTER(a), r, (size_t)n * sizeof *r);
}

typedef struct { VkCommandBuffer cb; VkBuffer s; VkImage d; VkImageLayout dl; uint32_t n; } A_CmdCopyBufferToImage;
static void R_CmdCopyBufferToImage(void *p) {
    A_CmdCopyBufferToImage *a = p;
    vkd_CmdCopyBufferToImage(a->cb, a->s, a->d, a->dl, a->n, (const VkBufferImageCopy *)AFTER(a));
}
VKAPI_ATTR void VKAPI_CALL vkq_CmdCopyBufferToImage(VkCommandBuffer cb, VkBuffer s, VkImage d, VkImageLayout dl,
                                                    uint32_t n, const VkBufferImageCopy *r) {
    WORK(cb);
    if (!vkq_on) { vkd_CmdCopyBufferToImage(cb, s, d, dl, n, r); return; }
    A_CmdCopyBufferToImage *a = push(R_CmdCopyBufferToImage, sizeof *a + ARR(VkBufferImageCopy, n));
    a->cb = cb; a->s = s; a->d = d; a->dl = dl; a->n = n;
    memcpy(AFTER(a), r, (size_t)n * sizeof *r);
}

typedef struct { VkCommandBuffer cb; VkImage s; VkImageLayout sl; VkBuffer d; uint32_t n; } A_CmdCopyImageToBuffer;
static void R_CmdCopyImageToBuffer(void *p) {
    A_CmdCopyImageToBuffer *a = p;
    vkd_CmdCopyImageToBuffer(a->cb, a->s, a->sl, a->d, a->n, (const VkBufferImageCopy *)AFTER(a));
}
VKAPI_ATTR void VKAPI_CALL vkq_CmdCopyImageToBuffer(VkCommandBuffer cb, VkImage s, VkImageLayout sl, VkBuffer d,
                                                    uint32_t n, const VkBufferImageCopy *r) {
    WORK(cb);
    if (!vkq_on) { vkd_CmdCopyImageToBuffer(cb, s, sl, d, n, r); return; }
    A_CmdCopyImageToBuffer *a = push(R_CmdCopyImageToBuffer, sizeof *a + ARR(VkBufferImageCopy, n));
    a->cb = cb; a->s = s; a->sl = sl; a->d = d; a->n = n;
    memcpy(AFTER(a), r, (size_t)n * sizeof *r);
}

#define IMAGE_TO_IMAGE(name, R)                                                                                       \
    typedef struct { VkCommandBuffer cb; VkImage s; VkImageLayout sl; VkImage d; VkImageLayout dl; uint32_t n; } A_##name; \
    static void R_##name(void *p) { A_##name *a = p; vkd_##name(a->cb, a->s, a->sl, a->d, a->dl, a->n, (const R *)AFTER(a)); } \
    VKAPI_ATTR void VKAPI_CALL vkq_##name(VkCommandBuffer cb, VkImage s, VkImageLayout sl, VkImage d, VkImageLayout dl, \
                                          uint32_t n, const R *r) {                                                   \
        WORK(cb);                                                                                                     \
        if (!vkq_on) { vkd_##name(cb, s, sl, d, dl, n, r); return; }                                                  \
        A_##name *a = push(R_##name, sizeof *a + ARR(R, n));                                                          \
        a->cb = cb; a->s = s; a->sl = sl; a->d = d; a->dl = dl; a->n = n;                                             \
        memcpy(AFTER(a), r, (size_t)n * sizeof *r);                                                                   \
    }
IMAGE_TO_IMAGE(CmdCopyImage, VkImageCopy)
IMAGE_TO_IMAGE(CmdResolveImage, VkImageResolve)

typedef struct { VkCommandBuffer cb; VkImage s; VkImageLayout sl; VkImage d; VkImageLayout dl; uint32_t n; VkFilter f; } A_CmdBlitImage;
static void R_CmdBlitImage(void *p) {
    A_CmdBlitImage *a = p;
    vkd_CmdBlitImage(a->cb, a->s, a->sl, a->d, a->dl, a->n, (const VkImageBlit *)AFTER(a), a->f);
}
VKAPI_ATTR void VKAPI_CALL vkq_CmdBlitImage(VkCommandBuffer cb, VkImage s, VkImageLayout sl, VkImage d, VkImageLayout dl,
                                            uint32_t n, const VkImageBlit *r, VkFilter f) {
    WORK(cb);
    if (!vkq_on) { vkd_CmdBlitImage(cb, s, sl, d, dl, n, r, f); return; }
    A_CmdBlitImage *a = push(R_CmdBlitImage, sizeof *a + ARR(VkImageBlit, n));
    a->cb = cb; a->s = s; a->sl = sl; a->d = d; a->dl = dl; a->n = n; a->f = f;
    memcpy(AFTER(a), r, (size_t)n * sizeof *r);
}

typedef struct { VkCommandBuffer cb; VkImage i; VkImageLayout l; VkClearColorValue c; uint32_t n; } A_CmdClearColorImage;
static void R_CmdClearColorImage(void *p) {
    A_CmdClearColorImage *a = p;
    vkd_CmdClearColorImage(a->cb, a->i, a->l, &a->c, a->n, (const VkImageSubresourceRange *)AFTER(a));
}
VKAPI_ATTR void VKAPI_CALL vkq_CmdClearColorImage(VkCommandBuffer cb, VkImage i, VkImageLayout l,
                                                  const VkClearColorValue *c, uint32_t n,
                                                  const VkImageSubresourceRange *r) {
    WORK(cb);
    if (!vkq_on) { vkd_CmdClearColorImage(cb, i, l, c, n, r); return; }
    A_CmdClearColorImage *a = push(R_CmdClearColorImage, sizeof *a + ARR(VkImageSubresourceRange, n));
    a->cb = cb; a->i = i; a->l = l; a->c = *c; a->n = n;
    memcpy(AFTER(a), r, (size_t)n * sizeof *r);
}

typedef struct { VkCommandBuffer cb; VkImage i; VkImageLayout l; VkClearDepthStencilValue v; uint32_t n; } A_CmdClearDepthStencilImage;
static void R_CmdClearDepthStencilImage(void *p) {
    A_CmdClearDepthStencilImage *a = p;
    vkd_CmdClearDepthStencilImage(a->cb, a->i, a->l, &a->v, a->n, (const VkImageSubresourceRange *)AFTER(a));
}
VKAPI_ATTR void VKAPI_CALL vkq_CmdClearDepthStencilImage(VkCommandBuffer cb, VkImage i, VkImageLayout l,
                                                         const VkClearDepthStencilValue *v, uint32_t n,
                                                         const VkImageSubresourceRange *r) {
    WORK(cb);
    if (!vkq_on) { vkd_CmdClearDepthStencilImage(cb, i, l, v, n, r); return; }
    A_CmdClearDepthStencilImage *a = push(R_CmdClearDepthStencilImage, sizeof *a + ARR(VkImageSubresourceRange, n));
    a->cb = cb; a->i = i; a->l = l; a->v = *v; a->n = n;
    memcpy(AFTER(a), r, (size_t)n * sizeof *r);
}

typedef struct { VkCommandBuffer cb; uint32_t n, nr; } A_CmdClearAttachments;
static void R_CmdClearAttachments(void *p) {
    A_CmdClearAttachments *a = p;
    const VkClearAttachment *at = (const VkClearAttachment *)AFTER(a);
    vkd_CmdClearAttachments(a->cb, a->n, at, a->nr, (const VkClearRect *)((const uint8_t *)at + ARR(VkClearAttachment, a->n)));
}
VKAPI_ATTR void VKAPI_CALL vkq_CmdClearAttachments(VkCommandBuffer cb, uint32_t n, const VkClearAttachment *at,
                                                   uint32_t nr, const VkClearRect *r) {
    if (!vkq_on) { vkd_CmdClearAttachments(cb, n, at, nr, r); return; }
    A_CmdClearAttachments *a = push(R_CmdClearAttachments, sizeof *a + ARR(VkClearAttachment, n) + ARR(VkClearRect, nr));
    a->cb = cb; a->n = n; a->nr = nr;
    memcpy(AFTER(a), at, (size_t)n * sizeof *at);
    memcpy(AFTER(a) + ARR(VkClearAttachment, n), r, (size_t)nr * sizeof *r);
}

// ---- calls with structs that point to more (no pNext chains: one would be recorded in place, after a drain)
// A colour attachment may chain one VkAttachmentFeedbackLoopInfoEXT (exec.c begin_pass, BOTW's G-buffer passes every
// frame): kept as a bit per attachment and chained again at replay, so those passes don't drain the ring.
typedef struct { VkCommandBuffer cb; VkRenderingInfo ri; uint32_t has_depth, has_stencil, feedback; } A_CmdBeginRendering;
static void R_CmdBeginRendering(void *p) {
    A_CmdBeginRendering *a = p;
    VkRenderingAttachmentInfo *at = (VkRenderingAttachmentInfo *)AFTER(a);
    uint32_t n = a->ri.colorAttachmentCount;
    VkAttachmentFeedbackLoopInfoEXT fb[8];
    for (uint32_t i = 0; i < n && i < 8; i++) {
        at[i].pNext = NULL;
        if (!(a->feedback & (1u << i))) continue;
        fb[i] = (VkAttachmentFeedbackLoopInfoEXT){.sType = VK_STRUCTURE_TYPE_ATTACHMENT_FEEDBACK_LOOP_INFO_EXT, .feedbackLoopEnable = VK_TRUE};
        at[i].pNext = &fb[i];
    }
    a->ri.pColorAttachments = n ? at : NULL;
    a->ri.pDepthAttachment = a->has_depth ? &at[n] : NULL;
    a->ri.pStencilAttachment = a->has_stencil ? &at[n + 1] : NULL;
    vkd_CmdBeginRendering(a->cb, &a->ri);
}
VKAPI_ATTR void VKAPI_CALL vkq_CmdBeginRendering(VkCommandBuffer cb, const VkRenderingInfo *ri) {
    WORK(cb);
    bool chained = ri->pNext != NULL || ri->colorAttachmentCount > 8;
    uint32_t feedback = 0;
    for (uint32_t i = 0; i < ri->colorAttachmentCount && !chained; i++) {
        const VkAttachmentFeedbackLoopInfoEXT *f = ri->pColorAttachments[i].pNext;
        if (!f) continue;
        if (f->sType == VK_STRUCTURE_TYPE_ATTACHMENT_FEEDBACK_LOOP_INFO_EXT && !f->pNext) { if (f->feedbackLoopEnable) feedback |= 1u << i; }
        else chained = true;
    }
    if (!vkq_on || chained) { if (vkq_on) vkq_drain(); vkd_CmdBeginRendering(cb, ri); return; }
    uint32_t n = ri->colorAttachmentCount;
    A_CmdBeginRendering *a = push(R_CmdBeginRendering, sizeof *a + ARR(VkRenderingAttachmentInfo, n + 2));
    a->cb = cb;
    a->ri = *ri;
    VkRenderingAttachmentInfo *at = (VkRenderingAttachmentInfo *)AFTER(a);
    if (n) memcpy(at, ri->pColorAttachments, (size_t)n * sizeof *at);
    a->has_depth = ri->pDepthAttachment != NULL;
    a->has_stencil = ri->pStencilAttachment != NULL;
    a->feedback = feedback;
    if (a->has_depth) at[n] = *ri->pDepthAttachment;
    if (a->has_stencil) at[n + 1] = *ri->pStencilAttachment;
}

typedef struct { VkCommandBuffer cb; VkDependencyInfo di; } A_CmdPipelineBarrier2;
static void R_CmdPipelineBarrier2(void *p) {
    A_CmdPipelineBarrier2 *a = p;
    uint8_t *q = AFTER(a);
    a->di.pMemoryBarriers = (const VkMemoryBarrier2 *)q;
    q += ARR(VkMemoryBarrier2, a->di.memoryBarrierCount);
    a->di.pBufferMemoryBarriers = (const VkBufferMemoryBarrier2 *)q;
    q += ARR(VkBufferMemoryBarrier2, a->di.bufferMemoryBarrierCount);
    a->di.pImageMemoryBarriers = (const VkImageMemoryBarrier2 *)q;
    vkd_CmdPipelineBarrier2(a->cb, &a->di);
}
VKAPI_ATTR void VKAPI_CALL vkq_CmdPipelineBarrier2(VkCommandBuffer cb, const VkDependencyInfo *di) {
    WORK(cb);
    bool chained = di->pNext != NULL;
    for (uint32_t i = 0; i < di->imageMemoryBarrierCount && !chained; i++) chained = di->pImageMemoryBarriers[i].pNext != NULL;
    for (uint32_t i = 0; i < di->bufferMemoryBarrierCount && !chained; i++) chained = di->pBufferMemoryBarriers[i].pNext != NULL;
    for (uint32_t i = 0; i < di->memoryBarrierCount && !chained; i++) chained = di->pMemoryBarriers[i].pNext != NULL;
    if (!vkq_on || chained) { if (vkq_on) vkq_drain(); vkd_CmdPipelineBarrier2(cb, di); return; }
    uint32_t nm = di->memoryBarrierCount, nb = di->bufferMemoryBarrierCount, ni = di->imageMemoryBarrierCount;
    A_CmdPipelineBarrier2 *a = push(R_CmdPipelineBarrier2, sizeof *a + ARR(VkMemoryBarrier2, nm) +
                                                               ARR(VkBufferMemoryBarrier2, nb) + ARR(VkImageMemoryBarrier2, ni));
    a->cb = cb;
    a->di = *di;
    uint8_t *q = AFTER(a);
    if (nm) memcpy(q, di->pMemoryBarriers, (size_t)nm * sizeof *di->pMemoryBarriers);
    q += ARR(VkMemoryBarrier2, nm);
    if (nb) memcpy(q, di->pBufferMemoryBarriers, (size_t)nb * sizeof *di->pBufferMemoryBarriers);
    q += ARR(VkBufferMemoryBarrier2, nb);
    if (ni) memcpy(q, di->pImageMemoryBarriers, (size_t)ni * sizeof *di->pImageMemoryBarriers);
}

typedef struct { VkCommandBuffer cb; VkCommandBufferBeginInfo bi; } A_BeginCommandBuffer;
static void R_BeginCommandBuffer(void *p) {
    A_BeginCommandBuffer *a = p;
    VkResult r = vkd_BeginCommandBuffer(a->cb, &a->bi);
    if (r != VK_SUCCESS) { LOG("[vk] deferred vkBeginCommandBuffer: %d\n", r); atomic_store(&failed, 1); }
}
VKAPI_ATTR VkResult VKAPI_CALL vkq_BeginCommandBuffer(VkCommandBuffer cb, const VkCommandBufferBeginInfo *bi) {
    if (!vkq_on || bi->pNext || bi->pInheritanceInfo) { if (vkq_on) vkq_drain(); return vkd_BeginCommandBuffer(cb, bi); }
    A_BeginCommandBuffer *a = push(R_BeginCommandBuffer, sizeof *a);
    a->cb = cb;
    a->bi = *bi;
    return VK_SUCCESS;
}

typedef struct { VkCommandBuffer cb; } A_EndCommandBuffer;
static void R_EndCommandBuffer(void *p) {
    A_EndCommandBuffer *a = p;
    VkResult r = vkd_EndCommandBuffer(a->cb);
    if (r != VK_SUCCESS) { LOG("[vk] deferred vkEndCommandBuffer: %d\n", r); atomic_store(&failed, 1); }
}
VKAPI_ATTR VkResult VKAPI_CALL vkq_EndCommandBuffer(VkCommandBuffer cb) {
    if (!vkq_on) return vkd_EndCommandBuffer(cb);
    A_EndCommandBuffer *a = push(R_EndCommandBuffer, sizeof *a);
    a->cb = cb;
    return VK_SUCCESS;
}

// Submissions: one VkSubmitInfo2 (what vk_submit_extra makes), no fence; anything else is submitted in place.
typedef struct { VkQueue q; VkSubmitInfo2 si; } A_QueueSubmit2;
static void R_QueueSubmit2(void *p) {
    A_QueueSubmit2 *a = p;
    uint8_t *q = AFTER(a);
    a->si.pWaitSemaphoreInfos = (const VkSemaphoreSubmitInfo *)q;
    q += ARR(VkSemaphoreSubmitInfo, a->si.waitSemaphoreInfoCount);
    a->si.pCommandBufferInfos = (const VkCommandBufferSubmitInfo *)q;
    q += ARR(VkCommandBufferSubmitInfo, a->si.commandBufferInfoCount);
    a->si.pSignalSemaphoreInfos = (const VkSemaphoreSubmitInfo *)q;
    vk_timing_driver(1, &a->si, false);
    uint64_t t0 = vkq_probe ? plat_now_ns() : 0;
    VkResult r = vkd_QueueSubmit2(a->q, 1, &a->si, VK_NULL_HANDLE);
    if (t0) atomic_fetch_add_explicit(&vkq_submit_ns, plat_now_ns() - t0, memory_order_relaxed);
    vk_timing_driver(1, &a->si, true);
    if (r != VK_SUCCESS) {
        LOG("[vk] deferred vkQueueSubmit2: %d\n", r);
        if (r == VK_ERROR_DEVICE_LOST) { extern void vk_devlost_dump(void); vk_devlost_dump(); }
        atomic_store(&failed, 1);
    }
}
VKAPI_ATTR VkResult VKAPI_CALL vkq_QueueSubmit2(VkQueue q, uint32_t n, const VkSubmitInfo2 *si, VkFence f) {
    if (!vkq_on || n != 1 || f || si->pNext) {
        if (vkq_on) vkq_drain();
        vk_timing_driver(n, si, false);
        VkResult r = vkd_QueueSubmit2(q, n, si, f);
        vk_timing_driver(n, si, true);
        return r;
    }
    uint32_t nw = si->waitSemaphoreInfoCount, nc = si->commandBufferInfoCount, ns = si->signalSemaphoreInfoCount;
    A_QueueSubmit2 *a = push(R_QueueSubmit2, sizeof *a + ARR(VkSemaphoreSubmitInfo, nw) +
                                                 ARR(VkCommandBufferSubmitInfo, nc) + ARR(VkSemaphoreSubmitInfo, ns));
    a->q = q;
    a->si = *si;
    uint8_t *p = AFTER(a);
    if (nw) memcpy(p, si->pWaitSemaphoreInfos, (size_t)nw * sizeof *si->pWaitSemaphoreInfos);
    p += ARR(VkSemaphoreSubmitInfo, nw);
    if (nc) memcpy(p, si->pCommandBufferInfos, (size_t)nc * sizeof *si->pCommandBufferInfos);
    p += ARR(VkCommandBufferSubmitInfo, nc);
    if (ns) memcpy(p, si->pSignalSemaphoreInfos, (size_t)ns * sizeof *si->pSignalSemaphoreInfos);
    publish();  // (the GPU starts on it as soon as the recorder gets there)
    return VK_SUCCESS;
}

// A wait for the GPU: what it waits for must have been submitted first.
VKAPI_ATTR VkResult VKAPI_CALL vkq_WaitSemaphores(VkDevice d, const VkSemaphoreWaitInfo *wi, uint64_t timeout) {
    if (vkq_on) vkq_drain();
    return vkd_WaitSemaphores(d, wi, timeout);
}
