#include <stdatomic.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "vk.h"
#include "timing.h"
#include "../../platform.h"
#ifdef _WIN32
#include <windows.h>
#endif

// Matches the submission ring in exec.c. Retire only after that slot's timeline
// value completed; reading queries adds no new GPU wait and no observer thread.
#define TIMING_SLOTS 32
bool vk_timing_on;
static bool tried;
static bool timing_log_on;
static VkQueryPool pool;
static VkQueryPool pass_pool;
static uint64_t mask;
static uint64_t previous_gpu_end, previous_seq;
#define PASS_MARKS 512
static struct {
    unsigned n, dropped;
    struct { char label[112]; unsigned draws; uint64_t shader, frame; } marks[PASS_MARKS];
} passes[TIMING_SLOTS];
uint64_t renderer_present_count(void);
static struct {
    uint64_t seq, enqueue, frame;
    unsigned reasons, buffers;
    _Atomic uint64_t driver, returned, driver_seq;
} timing[TIMING_SLOTS];

// Diagnostic only, on the producer. Keep raw device ticks in the log and publish
// a fresh correspondence once a second. Consumers must account for max deviation
// and timestampValidBits; subtracting independently based trace clocks is invalid.
static void timing_calibrate(void) {
    static bool initialized;
    static PFN_vkGetCalibratedTimestampsKHR get;
    static uint64_t last, frequency = 1000000000ull;
    static VkTimeDomainKHR host_domain;
    if (!g_vk.calibrated_time) return;
    if (!initialized) {
        initialized = true;
#ifdef _WIN32
        LARGE_INTEGER f;
        if (!QueryPerformanceFrequency(&f) || f.QuadPart <= 0) return;
        frequency = (uint64_t)f.QuadPart;
        host_domain = VK_TIME_DOMAIN_QUERY_PERFORMANCE_COUNTER_KHR;
#else
        host_domain = VK_TIME_DOMAIN_CLOCK_MONOTONIC_KHR;  // plat_now_ns on Linux
#endif
        PFN_vkGetPhysicalDeviceCalibrateableTimeDomainsKHR domains =
            (PFN_vkGetPhysicalDeviceCalibrateableTimeDomainsKHR)vkGetInstanceProcAddr(g_vk.inst,
                g_vk.calibrated_time == 1 ? "vkGetPhysicalDeviceCalibrateableTimeDomainsKHR" : "vkGetPhysicalDeviceCalibrateableTimeDomainsEXT");
        if (!domains) return;
        uint32_t n = 0;
        if (domains(g_vk.phys, &n, NULL) != VK_SUCCESS || !n || n > 32) return;
        VkTimeDomainKHR values[32];
        if (domains(g_vk.phys, &n, values) != VK_SUCCESS) return;
        bool device_found = false, host_found = false;
        for (uint32_t i = 0; i < n; i++) {
            device_found |= values[i] == VK_TIME_DOMAIN_DEVICE_KHR;
            host_found |= values[i] == host_domain;
        }
        if (!device_found || !host_found) return;
        get = (PFN_vkGetCalibratedTimestampsKHR)vkGetDeviceProcAddr(g_vk.dev,
            g_vk.calibrated_time == 1 ? "vkGetCalibratedTimestampsKHR" : "vkGetCalibratedTimestampsEXT");
    }
    if (!get) return;
    uint64_t now = plat_now_ns();
    if (last && now - last < 1000000000ull) return;
    last = now;
    VkCalibratedTimestampInfoKHR info[2] = {
        {.sType = VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_KHR, .timeDomain = VK_TIME_DOMAIN_DEVICE_KHR},
        {.sType = VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_KHR, .timeDomain = host_domain}};
    uint64_t best[2] = {0}, deviation = UINT64_MAX;
    for (int i = 0; i < 3; i++) {
        uint64_t t[2], d;
        if (get(g_vk.dev, 2, info, t, &d) == VK_SUCCESS && d < deviation) {
            best[0] = t[0]; best[1] = t[1]; deviation = d;
        }
    }
    if (deviation == UINT64_MAX) return;
    uint64_t host = (uint64_t)((__uint128_t)best[1] * 1000000000ull / frequency);
    TRACE_LOG("[vkclock] {\"cpu_ns\":%llu,\"gpu_tick\":%llu,\"max_deviation_ns\":%llu,\"period_ns\":%.9f,\"valid_mask\":%llu}\n",
        (unsigned long long)host, (unsigned long long)(best[0] & mask), (unsigned long long)deviation,
        g_vk.props.limits.timestampPeriod, (unsigned long long)mask);
}

static void timing_init(void) {
    if (tried) return;
    tried = true;
    vk_frame_trace_init();
    const char *e = getenv("SAVAGEWIND_VK_TIMING");
    const char *f = getenv("SAVAGEWIND_FRAME_LOG");
    timing_log_on = e ? atoi(e) != 0 : f && atoi(f) >= 2;
    if (!vk_frame_trace_on && !timing_log_on) return;
    uint32_t n = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(g_vk.phys, &n, NULL);
    VkQueueFamilyProperties *q = calloc(n, sizeof *q);
    if (!q) return;
    vkGetPhysicalDeviceQueueFamilyProperties(g_vk.phys, &n, q);
    unsigned bits = g_vk.qfam < n ? q[g_vk.qfam].timestampValidBits : 0;
    free(q);
    if (!bits) return;
    mask = bits == 64 ? UINT64_MAX : (1ull << bits) - 1;
    VkQueryPoolCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
        .queryType = VK_QUERY_TYPE_TIMESTAMP, .queryCount = TIMING_SLOTS * 4};
    if (vkCreateQueryPool(g_vk.dev, &ci, NULL, &pool) != VK_SUCCESS) return;
    if (vk_frame_trace_on) {
        ci.queryCount = TIMING_SLOTS * PASS_MARKS;
        if (vkCreateQueryPool(g_vk.dev, &ci, NULL, &pass_pool) != VK_SUCCESS)
            LOG("[frame-trace] GPU pass query pool unavailable; CPU/submission trace only\n");
    }
    vk_timing_on = true;
    TRACE_LOG("[vktrace] timestamp_period_ns=%.6f valid_bits=%u. GPU spans include uploads, stalls and overlap; they are not utilization. CPU and GPU clocks are separate.\n",
        g_vk.props.limits.timestampPeriod, bits);
}

void vk_timing_begin(VkCommandBuffer cb, unsigned slot, bool upload) {
    timing_init();
    if (!vk_timing_on) return;
    uint32_t first = slot * 4 + (upload ? 0 : 2);
    vkCmdResetQueryPool(cb, pool, first, 2);
    vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, pool, first);
    timing[slot].buffers |= upload ? 1 : 2;
    if (!upload && pass_pool) {
        passes[slot].n = passes[slot].dropped = 0;
        vkCmdResetQueryPool(cb, pass_pool, slot * PASS_MARKS, PASS_MARKS);
    }
}

void vk_timing_end(VkCommandBuffer cb, unsigned slot, bool upload) {
    if (!upload && pass_pool) vk_timing_mark(cb, slot, "submission end");
    if (vk_timing_on)
        vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, pool, slot * 4 + (upload ? 1 : 3));
}

void vk_timing_enqueue(unsigned slot, uint64_t seq, unsigned reasons) {
    if (!vk_timing_on) return;
    timing_calibrate();
    timing[slot].seq = seq;
    timing[slot].frame = renderer_present_count();
    timing[slot].reasons = reasons;
    timing[slot].enqueue = plat_now_ns();
}

void vk_timing_mark(VkCommandBuffer cb, unsigned slot, const char *label) {
    if (!pass_pool) return;
    unsigned n = passes[slot].n;
    // Keep one closing query even when a very large submission exceeds the cap.
    bool last = !strcmp(label, "submission end");
    if (n >= PASS_MARKS - (last ? 0 : 1)) { passes[slot].dropped++; return; }
    snprintf(passes[slot].marks[n].label, sizeof passes[slot].marks[n].label, "%s", label);
    passes[slot].marks[n].frame = renderer_present_count();
    passes[slot].marks[n].draws = 0;
    passes[slot].marks[n].shader = 0;
    vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, pass_pool, slot * PASS_MARKS + n);
    passes[slot].n++;
}
void vk_timing_draw(unsigned slot, uint64_t shader) {
    if (!pass_pool || !passes[slot].n) return;
    unsigned n = passes[slot].n - 1;
    if (!passes[slot].marks[n].draws++) passes[slot].marks[n].shader = shader;
}

// Called around the actual driver call, including when the recorder defers it.
// Publish both host readings together. A very fast GPU may complete before the
// API returns; retire then omits host readings instead of using an old slot.
void vk_timing_driver(uint32_t n, const VkSubmitInfo2 *si, bool returned) {
    if (!vk_timing_on) return;
    uint64_t now = plat_now_ns();
    for (uint32_t i = 0; i < n; i++) for (uint32_t j = 0; j < si[i].signalSemaphoreInfoCount; j++) {
        const VkSemaphoreSubmitInfo *s = &si[i].pSignalSemaphoreInfos[j];
        if (s->semaphore != g_vk.timeline || !s->value) continue;
        unsigned k = (unsigned)((s->value - 1) % TIMING_SLOTS);
        if (!returned) atomic_store_explicit(&timing[k].driver, now, memory_order_relaxed);
        else {
            atomic_store_explicit(&timing[k].returned, now, memory_order_relaxed);
            atomic_store_explicit(&timing[k].driver_seq, s->value, memory_order_release);
        }
    }
}

void vk_timing_retire(unsigned slot) {
    if (!vk_timing_on || !timing[slot].seq) return;
    uint64_t t[4] = {0}, seq = timing[slot].seq;
    unsigned first = timing[slot].buffers & 1 ? 0 : 2;
    unsigned count = 4 - first;
    if (vkGetQueryPoolResults(g_vk.dev, pool, slot * 4 + first, count,
                             count * sizeof(uint64_t), t + first, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
        double period = g_vk.props.limits.timestampPeriod / 1e6;
        uint64_t driver = 0, returned = 0;
        if (atomic_load_explicit(&timing[slot].driver_seq, memory_order_acquire) == seq) {
            driver = atomic_load_explicit(&timing[slot].driver, memory_order_relaxed);
            returned = atomic_load_explicit(&timing[slot].returned, memory_order_relaxed);
        }
        if (vk_frame_trace_on) {
            if (driver >= timing[slot].enqueue && returned >= driver) {
                vk_frame_trace_event("Recorder queue delay", 2, timing[slot].frame, seq, timing[slot].enqueue, driver, 0, 0);
                vk_frame_trace_event("vkQueueSubmit2", 2, timing[slot].frame, seq, driver, returned, 0, 0);
            }
            double ns = g_vk.props.limits.timestampPeriod;
            uint64_t begin = (uint64_t)((t[first] & mask) * ns);
            uint64_t end = begin + (uint64_t)(((t[3] - t[first]) & mask) * ns);
            if (previous_seq + 1 == seq && previous_gpu_end && begin > previous_gpu_end)
                vk_frame_trace_event("Interval between GPU submissions (possible idle)", 12, timing[slot].frame, seq,
                    previous_gpu_end, begin, 0, 0);
            previous_gpu_end = end;
            previous_seq = seq;
            vk_frame_trace_event("GPU submission (includes uploads/stalls)", 11, timing[slot].frame, seq,
                begin, end, 0, 0);
            if (!first) vk_frame_trace_event("Upload command buffer", 11, timing[slot].frame, seq,
                begin, begin + (uint64_t)(((t[1] - t[0]) & mask) * ns), 0, 0);
        }
        if (timing_log_on) TRACE_LOG("[vktrace] {\"seq\":%llu,\"frame\":%llu,\"enqueue_ns\":%llu,\"driver_ns\":%llu,\"return_ns\":%llu,"
            "\"gpu_begin_tick\":%llu,\"gpu_end_tick\":%llu,\"gpu_ms\":%.6f,\"main_ms\":%.6f,\"upload_ms\":%.6f,\"reasons\":%u}\n",
            (unsigned long long)seq, (unsigned long long)timing[slot].frame, (unsigned long long)timing[slot].enqueue,
            (unsigned long long)driver, (unsigned long long)returned,
            (unsigned long long)(t[first] & mask), (unsigned long long)(t[3] & mask),
            ((t[3] - t[first]) & mask) * period, ((t[3] - t[2]) & mask) * period,
            first ? 0.0 : ((t[1] - t[0]) & mask) * period, timing[slot].reasons);
    }
    if (pass_pool && passes[slot].n > 1) {
        uint64_t ticks[PASS_MARKS];
        unsigned n = passes[slot].n;
        if (vkGetQueryPoolResults(g_vk.dev, pass_pool, slot * PASS_MARKS, n, sizeof ticks, ticks, sizeof(uint64_t),
                                 VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
            double ns = g_vk.props.limits.timestampPeriod;
            for (unsigned i = 0; i + 1 < n; i++) {
                uint64_t begin = (uint64_t)((ticks[i] & mask) * ns);
                vk_frame_trace_event(passes[slot].marks[i].label, 10, passes[slot].marks[i].frame, seq, begin,
                    begin + (uint64_t)(((ticks[i + 1] - ticks[i]) & mask) * ns),
                    passes[slot].marks[i].draws, passes[slot].marks[i].shader);
            }
        }
        if (passes[slot].dropped) LOG("[frame-trace] seq %llu: %u pass marks omitted (capacity)\n",
            (unsigned long long)seq, passes[slot].dropped);
    }
    timing[slot].seq = 0;
    timing[slot].buffers = 0;
}
