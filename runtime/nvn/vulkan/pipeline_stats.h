#pragma once
#include "vk.h"

// Opt-in compiler diagnostics, not runtime GPU utilization or a throughput mode.
VkPipelineCreateFlags vk_pipeline_stats_flags(void);
void vk_pipeline_stats(VkPipeline pipeline, const char *kind, uint64_t id0, uint64_t id1,
                       uint64_t state_key, const uint32_t local[3]);
