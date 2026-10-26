#pragma once
#include <stdbool.h>
#include <stdint.h>
#include <stdatomic.h>
// Explicitly destructive diagnostics. No mode is enabled without SAVAGEWIND_SURVEY.
enum { SURVEY_OFF, SURVEY_MAIN, SURVEY_JOBS, SURVEY_PRODUCER, SURVEY_DESCRIPTOR,
       SURVEY_RECORDER, SURVEY_GPU, SURVEY_STUB, SURVEY_CPU_FENCE, SURVEY_OVERLAP_SENSITIVITY, SURVEY_WAIT_JOBS, SURVEY_QUEUE_AHEAD, SURVEY_DOWNSTREAM, SURVEY_JOB_WINDOW, SURVEY_FENCE_FLAGS, SURVEY_SAFE_OVERLAP };
extern int perf_survey_mode;
extern _Atomic uint64_t perf_survey_frame;
extern _Atomic int perf_survey_block, perf_survey_enabled;
extern uint64_t perf_survey_epoch, perf_survey_start_ns;
void perf_survey_init(void);
void perf_survey_present(void);
void perf_survey_delay(int role, unsigned identity);
struct KThread;
void perf_survey_guest(struct KThread *t);

// Optimizations on trial, one bit each (PERFX_*): SAVAGEWIND_PERFX=<mask> turns them on. With
// SAVAGEWIND_PERFX_AB=<frames> they go off / on / on / off in blocks of that many presented frames from
// SAVAGEWIND_PERFX_AB_START (default 3000), for an A/B within one launch: "[perfx] frame <n> block <b> on|off".
// The render thread takes the bits on at the start of each command list (perfx_list_begin; the list's draw jobs are
// all done before the next list starts), so a list's translation and its workers see one state.
enum { PERFX_CBR = 1, PERFX_TEXBIND = 2, PERFX_SMT = 4, PERFX_MATUP = 8, PERFX_KSPIN = 16, PERFX_GWORDER = 32, PERFX_DJSPIN = 64, PERFX_MICRO = 128, PERFX_DSITE = 256 };
extern _Atomic uint32_t g_perfx, g_perfx_list;
extern uint32_t g_perfx_mask;  // the bits on trial (SAVAGEWIND_PERFX)
void perfx_list_begin(void);
static inline bool perfx(uint32_t bit) { return atomic_load_explicit(&g_perfx_list, memory_order_relaxed) & bit; }
static inline bool perfx_trial(uint32_t bit) { return g_perfx_mask & bit; }
static inline bool perf_survey_on(int mode) {
    if (perf_survey_mode == SURVEY_DOWNSTREAM || perf_survey_mode == SURVEY_JOB_WINDOW) {
        int b = atomic_load_explicit(&perf_survey_block, memory_order_relaxed);
        if (mode == SURVEY_CPU_FENCE) return b >= (perf_survey_mode == SURVEY_JOB_WINDOW ? 4 : 0);
        int role = perf_survey_mode == SURVEY_JOB_WINDOW ? SURVEY_JOB_WINDOW : b < 4 ? SURVEY_RECORDER : SURVEY_DESCRIPTOR;
        return mode == role && atomic_load_explicit(&perf_survey_enabled, memory_order_relaxed);
    }
    if (perf_survey_mode == SURVEY_WAIT_JOBS || perf_survey_mode == SURVEY_QUEUE_AHEAD) {
        int b = atomic_load_explicit(&perf_survey_block, memory_order_relaxed);
        if (mode == SURVEY_CPU_FENCE) return b >= 0;
        int role = perf_survey_mode == SURVEY_QUEUE_AHEAD ? SURVEY_QUEUE_AHEAD : b < 8 ? SURVEY_JOBS : SURVEY_STUB;
        return mode == role && atomic_load_explicit(&perf_survey_enabled, memory_order_relaxed);
    }
    if (perf_survey_mode == SURVEY_OVERLAP_SENSITIVITY) {
        int b = atomic_load_explicit(&perf_survey_block, memory_order_relaxed);
        if (mode == SURVEY_CPU_FENCE) return b >= 0; // constant throughout BOTH measurement cycles
        int role = b < 4 ? SURVEY_PRODUCER : b < 8 ? SURVEY_MAIN : SURVEY_JOBS;
        return mode == role && atomic_load_explicit(&perf_survey_enabled, memory_order_relaxed);
    }
    return perf_survey_mode == mode && atomic_load_explicit(&perf_survey_enabled, memory_order_relaxed);
}
