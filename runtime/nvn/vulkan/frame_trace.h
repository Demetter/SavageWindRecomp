#pragma once
#include <stdbool.h>
#include <stdint.h>

// Opt-in bounded flight recorder. CPU and GPU events have separate clocks;
// frame/submission IDs correlate them without pretending those clocks align.
extern bool vk_frame_trace_on;
void vk_frame_trace_init(void);
void vk_frame_trace_event(const char *name, unsigned lane, uint64_t frame, uint64_t seq,
                          uint64_t begin_ns, uint64_t end_ns, unsigned draws, uint64_t shader);
void vk_frame_trace_present(uint64_t frame, int target);
void vk_frame_trace_phase(const char *name, uint64_t frame, uint64_t seq, uint64_t begin_ns, uint64_t end_ns,
                          unsigned draws, uint64_t scheduled_cycles, uint32_t host_thread_id);
void vk_frame_trace_reset(void);
