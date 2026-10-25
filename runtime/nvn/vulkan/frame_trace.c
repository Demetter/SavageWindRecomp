// SAVAGEWIND_FRAME_TRACE=<output prefix> ("1": build/frame_trace).
// Detect small outliers against the preceding 64 uncapped frames, keep their
// neighbors, and write on a background thread. No GPU drain or per-event I/O.
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#endif
#include "frame_trace.h"
#include "../../rt.h"
#include "../../hle.h"
#include "../../platform.h"

#define EVENTS 65536
#define HISTORY 64
#define BEFORE 8
#define AFTER 8
#define RETIRE_GRACE 8
typedef struct {
    uint64_t frame, seq, begin, end, shader;
    uint64_t scheduled_cycles;
    uint32_t host_thread_id;
    unsigned lane, draws;
    char name[112];
} Event;
typedef struct {
    Event *events;
    size_t n;
    uint64_t frame, lost;
    double baseline_ms, trigger_ms;
    unsigned number;
    bool large;
    char path[1024];
} Capture;
bool vk_frame_trace_on;
static Event *events;
static uint64_t put, last_present, started, trigger_frame, cooldown_frame;
static double history[HISTORY], trigger_ms, baseline_ms;
static unsigned history_n, history_put, captures, limit = 4;
static unsigned micro_captures, large_captures, large_limit = 4;
static bool trigger_large, forced_done;
static double min_delta_ms = 0.5, relative_delta = 0.20, start_s = 30;
static double large_ms = 16.667, large_ratio = 2.0;
static uint64_t at_frame;
static char prefix[896];
static _Atomic bool writing;
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t writer_cv = PTHREAD_COND_INITIALIZER;
static Capture saved_capture;
static void *capture_writer(void *arg);

static double option(const char *name, double def, double lo, double hi) {
    const char *v = getenv(name);
    if (!v) return def;
    char *end;
    double x = strtod(v, &end);
    return end != v && !*end && isfinite(x) && x >= lo && x <= hi ? x : def;
}
void vk_frame_trace_init(void) {
    const char *v = getenv("SAVAGEWIND_FRAME_TRACE");
    if (!v || !*v || !strcmp(v, "0")) return;
    events = calloc(EVENTS, sizeof *events);
    if (!events) return;
    saved_capture.events = calloc(EVENTS, sizeof *events);
    if (!saved_capture.events) { free(events); events = NULL; return; }
    // Commit both buffers during startup, not while holding the trace lock in
    // the present loop after a hitch. One writer owns the saved buffer at a time.
    for (size_t i = 0; i < EVENTS * sizeof(Event); i += 4096) {
        ((volatile unsigned char *)events)[i] = 0;
        ((volatile unsigned char *)saved_capture.events)[i] = 0;
    }
    pthread_t thread;
    if (pthread_create(&thread, NULL, capture_writer, NULL)) {
        free(events); free(saved_capture.events); events = saved_capture.events = NULL;
        LOG("[frame-trace] writer could not start; tracing disabled\n"); return;
    }
    pthread_detach(thread);
    snprintf(prefix, sizeof prefix, "%s", !strcmp(v, "1") ? "build/frame_trace" : v);
    min_delta_ms = option("SAVAGEWIND_FRAME_TRACE_DELTA_MS", 0.5, 0.05, 1000);
    relative_delta = option("SAVAGEWIND_FRAME_TRACE_RATIO", 0.20, 0, 100);
    start_s = option("SAVAGEWIND_FRAME_TRACE_START_S", 30, 0, 86400);
    at_frame = (uint64_t)option("SAVAGEWIND_FRAME_TRACE_AT", 0, 0, 1e12);
    limit = (unsigned)option("SAVAGEWIND_FRAME_TRACE_COUNT", 4, 1, 100);
    large_limit = (unsigned)option("SAVAGEWIND_FRAME_TRACE_LARGE_COUNT", 4, 0, 100);
    large_ms = option("SAVAGEWIND_FRAME_TRACE_LARGE_MS", 16.667, 1, 1000);
    large_ratio = option("SAVAGEWIND_FRAME_TRACE_LARGE_RATIO", 2, 1, 100);
    started = plat_now_ns();
    vk_frame_trace_on = true;
    LOG("[frame-trace] rolling capture: median + max(%.3f ms, %.0f%%), after %.0f s; %u micro captures plus %u large captures "
        "(max %.3f ms, %.1fx median). Diagnostic overhead applies.\n",
        min_delta_ms, relative_delta * 100, start_s, limit, large_limit, large_ms, large_ratio);
}

static bool capture_budget(void) {
    return !forced_done && (micro_captures < limit || large_captures < large_limit);
}

static void add(const char *name, unsigned lane, uint64_t frame, uint64_t seq,
                uint64_t begin, uint64_t end, unsigned draws, uint64_t shader) {
    if (end < begin) return;
    Event *e = &events[put++ % EVENTS];
    *e = (Event){.frame = frame, .seq = seq, .begin = begin, .end = end,
                 .lane = lane, .draws = draws, .shader = shader};
    snprintf(e->name, sizeof e->name, "%s", name);
}
void vk_frame_trace_event(const char *name, unsigned lane, uint64_t frame, uint64_t seq,
                          uint64_t begin, uint64_t end, unsigned draws, uint64_t shader) {
    if (!vk_frame_trace_on) return;
    pthread_mutex_lock(&lock);
    if (capture_budget()) add(name, lane, frame, seq, begin, end, draws, shader);
    pthread_mutex_unlock(&lock);
}
void vk_frame_trace_phase(const char *name, uint64_t frame, uint64_t seq, uint64_t begin, uint64_t end,
                          unsigned draws, uint64_t cycles, uint32_t tid) {
    if (!vk_frame_trace_on || end < begin) return;
    pthread_mutex_lock(&lock);
    if (capture_budget()) {
        add(name, 6, frame, seq, begin, end, draws, 0);
        Event *e = &events[(put - 1) % EVENTS];
        e->scheduled_cycles = cycles;
        e->host_thread_id = tid;
    }
    pthread_mutex_unlock(&lock);
}

static void json_string(FILE *f, const char *s) {
    fputc('"', f);
    for (; *s; s++) {
        unsigned char c = (unsigned char)*s;
        if (c == '"' || c == '\\') fprintf(f, "\\%c", c);
        else if (c < 32) fprintf(f, "\\u%04x", c);
        else fputc(c, f);
    }
    fputc('"', f);
}
static void *write_capture(void *arg) {
    Capture *c = arg;
    FILE *f = fopen(c->path, "wb");
    if (f) {
        uint64_t base[2] = {UINT64_MAX, UINT64_MAX};
        for (size_t i = 0; i < c->n; i++) {
            Event *e = &c->events[i];
            unsigned domain = e->lane >= 10;
            if (e->begin < base[domain]) base[domain] = e->begin;
        }
        fprintf(f, "{\"displayTimeUnit\":\"ms\",\"metadata\":{\"trigger_frame\":%llu,\"trigger_ms\":%.6f,"
                "\"baseline_ms\":%.6f,\"overwritten_events\":%llu,\"diagnostic\":true,\"trigger_kind\":\"%s\","
                "\"cpu_origin_ns\":%llu,\"gpu_origin_ns\":%llu,"
                "\"clock_note\":\"CPU and GPU origins are independent; correlate by frame and seq. GPU spans include stalls, not utilization. Frame IDs associate queued work with a present, not scanout.\"},\"traceEvents\":[\n",
                (unsigned long long)c->frame, c->trigger_ms, c->baseline_ms, (unsigned long long)c->lost,
                at_frame ? "forced" : c->large ? "large" : "micro",
                (unsigned long long)base[0], (unsigned long long)base[1]);
        const char *names[] = {"Frame intervals", "Render translation", "Vulkan recorder", "GPU waits", "Resource and shader preparation", "Guest memory dependencies", "CPU command preparation between operation marks", "Guest queue delays", "Guest fence wait phases", "Presentation"};
        bool comma = false;
        for (unsigned i = 0; i < sizeof names / sizeof *names; i++) {
            fprintf(f, "%s{\"ph\":\"M\",\"pid\":1,\"tid\":%u,\"name\":\"thread_name\",\"args\":{\"name\":\"%s\"}}", comma ? ",\n" : "", i, names[i]);
            comma = true;
        }
        fprintf(f, ",\n{\"ph\":\"M\",\"pid\":2,\"tid\":10,\"name\":\"thread_name\",\"args\":{\"name\":\"GPU pass timestamp spans\"}}");
        fprintf(f, ",\n{\"ph\":\"M\",\"pid\":2,\"tid\":11,\"name\":\"thread_name\",\"args\":{\"name\":\"GPU submission timestamp spans\"}}");
        fprintf(f, ",\n{\"ph\":\"M\",\"pid\":2,\"tid\":12,\"name\":\"thread_name\",\"args\":{\"name\":\"Intervals between GPU submissions\"}}");
        for (size_t i = 0; i < c->n; i++) {
            Event *e = &c->events[i];
            unsigned domain = e->lane >= 10;
            fprintf(f, ",\n{\"ph\":\"X\",\"pid\":%u,\"tid\":%u,\"name\":", domain + 1, e->lane);
            json_string(f, e->name);
            fprintf(f, ",\"ts\":%.3f,\"dur\":%.3f,\"args\":{\"frame\":%llu,\"seq\":%llu,\"draws\":%u,\"first_shader\":\"%016llx\"",
                    (e->begin - base[domain]) / 1e3, (e->end - e->begin) / 1e3,
                    (unsigned long long)e->frame, (unsigned long long)e->seq, e->draws, (unsigned long long)e->shader);
            if (e->host_thread_id)
                fprintf(f, ",\"scheduled_cpu_cycles\":%llu,\"host_thread_id\":%u",
                        (unsigned long long)e->scheduled_cycles, e->host_thread_id);
            fputs("}}", f);
        }
        fprintf(f, "\n]}\n");
        bool failed = ferror(f) != 0;
        if (fclose(f)) failed = true;
        LOG("[frame-trace] %s %s (%zu events; frame %llu %.3f ms, preceding median %.3f ms)\n",
            failed ? "write failed:" : "saved", c->path, c->n, (unsigned long long)c->frame, c->trigger_ms, c->baseline_ms);
    } else LOG("[frame-trace] cannot write %s\n", c->path);
    atomic_store_explicit(&writing, false, memory_order_release);
    return NULL;
}
static void *capture_writer(void *arg) {
    (void)arg;
#ifdef _WIN32
    SetThreadPriority(GetCurrentThread(), THREAD_MODE_BACKGROUND_BEGIN);
#endif
    for (;;) {
        pthread_mutex_lock(&lock);
        while (!atomic_load_explicit(&writing, memory_order_acquire)) pthread_cond_wait(&writer_cv, &lock);
        pthread_mutex_unlock(&lock);
        write_capture(&saved_capture);
    }
    return NULL;
}
static int compare_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}
void vk_frame_trace_present(uint64_t frame, int target) {
    if (!vk_frame_trace_on) return;
    uint64_t now = plat_now_ns();
    Capture *c = NULL;
    pthread_mutex_lock(&lock);
    if (!capture_budget()) { pthread_mutex_unlock(&lock); return; }
    double ms = last_present ? (now - last_present) / 1e6 : 0;
    if (last_present) add("Frame interval", 0, frame, 0, last_present, now, 0, 0);
    last_present = now;
    if (target || !ms) { history_n = history_put = 0; }
    else {
        // Keep recording after the micro budget is spent. A large hitch has its
        // own budget and bypasses the micro cooldown. Keep a new pending trigger
        // even while the writer drains the previous snapshot. Within a cluster,
        // center the pending neighborhood on its worst hitch, not the first one.
        if (history_n >= 32 && (now - started) / 1e9 >= start_s &&
            ((!trigger_frame && frame > cooldown_frame) || (!at_frame && ms >= large_ms))) {
            double sorted[HISTORY];
            memcpy(sorted, history, history_n * sizeof *sorted);
            qsort(sorted, history_n, sizeof *sorted, compare_double);
            double median = (sorted[(history_n - 1) / 2] + sorted[history_n / 2]) * 0.5;
            double delta = fmax(min_delta_ms, median * relative_delta);
            bool large = ms >= fmax(large_ms, median * large_ratio);
            bool accept = !trigger_frame && (at_frame ? frame >= at_frame :
                (large ? large_captures < large_limit : micro_captures < limit && frame > cooldown_frame && ms > median + delta));
            bool upgrade = !at_frame && trigger_frame && large && large_captures < large_limit &&
                           (!trigger_large || ms > trigger_ms);
            if (accept || upgrade) {
                trigger_frame = frame;
                trigger_ms = ms;
                baseline_ms = median;
                trigger_large = large;
            }
        }
        history[history_put++ % HISTORY] = ms;
        if (history_n < HISTORY) history_n++;
    }
    // Give existing submission-slot retirement time to deliver the GPU queries.
    // Missing results are left missing; this recorder never forces a GPU wait.
    if (trigger_frame && frame >= trigger_frame + AFTER + RETIRE_GRACE && !atomic_load(&writing)) {
        c = &saved_capture;
        Event *storage = c->events;
        *c = (Capture){.events = storage};
        {
            uint64_t copy_begin = plat_now_ns();
            uint64_t lo = trigger_frame > BEFORE ? trigger_frame - BEFORE : 0, hi = trigger_frame + AFTER;
            uint64_t begin = put > EVENTS ? put - EVENTS : 0;
            for (uint64_t i = begin; i < put; i++) {
                Event *e = &events[i % EVENTS];
                if (e->frame >= lo && e->frame <= hi) c->events[c->n++] = *e;
            }
            // Whole-run overwrite count is explicit; the report also checks for
            // missing frame intervals and unmatched submission/pass coverage.
            c->lost = begin;
            c->frame = trigger_frame; c->trigger_ms = trigger_ms; c->baseline_ms = baseline_ms;
            c->large = trigger_large;
            if (trigger_large) large_captures++; else micro_captures++;
            c->number = ++captures;
            snprintf(c->path, sizeof c->path, "%s_%u.json", prefix, c->number);
            atomic_store(&writing, true);
            add("Trace snapshot copy (diagnostic)", 4, frame, 0, copy_begin, plat_now_ns(), 0, 0);
            pthread_cond_signal(&writer_cv);
        }
        cooldown_frame = frame + HISTORY;
        trigger_frame = 0;
        if (at_frame) forced_done = true;
    }
    pthread_mutex_unlock(&lock);
}
void vk_frame_trace_reset(void) {
    pthread_mutex_lock(&lock);
    put = last_present = trigger_frame = cooldown_frame = 0;
    trigger_large = false;
    history_n = history_put = 0;
    started = plat_now_ns();
    pthread_mutex_unlock(&lock);
}
RT_ON_RESET(frame_trace_reset) { vk_frame_trace_reset(); }
