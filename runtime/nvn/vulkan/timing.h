#pragma once
#include <stdbool.h>
#include <stdint.h>
#include <vulkan/vulkan.h>
#include "frame_trace.h"

// Diagnostic only. CPU queue timestamps and GPU timestamp spans are separate clocks.
extern bool vk_timing_on;
void vk_timing_begin(VkCommandBuffer cb, unsigned slot, bool upload);
void vk_timing_end(VkCommandBuffer cb, unsigned slot, bool upload);
void vk_timing_enqueue(unsigned slot, uint64_t seq, unsigned reasons);
void vk_timing_driver(uint32_t n, const VkSubmitInfo2 *si, bool returned);
void vk_timing_retire(unsigned slot);
void vk_timing_mark(VkCommandBuffer cb, unsigned slot, const char *label);
void vk_timing_draw(unsigned slot, uint64_t shader);
