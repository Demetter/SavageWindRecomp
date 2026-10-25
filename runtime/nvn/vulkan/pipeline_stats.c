#include <inttypes.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include "pipeline_stats.h"
#include "../../rt.h"

static pthread_once_t once = PTHREAD_ONCE_INIT;
static PFN_vkGetPipelineExecutablePropertiesKHR get_properties;
static PFN_vkGetPipelineExecutableStatisticsKHR get_statistics;

static void init(void) {
    get_properties = (PFN_vkGetPipelineExecutablePropertiesKHR)vkGetDeviceProcAddr(g_vk.dev, "vkGetPipelineExecutablePropertiesKHR");
    get_statistics = (PFN_vkGetPipelineExecutableStatisticsKHR)vkGetDeviceProcAddr(g_vk.dev, "vkGetPipelineExecutableStatisticsKHR");
    if (!get_properties || !get_statistics) LOG("[pipeline-exec] query entrypoints unavailable\n");
}

VkPipelineCreateFlags vk_pipeline_stats_flags(void) {
    return g_vk.pipeline_stats ? VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR : 0;
}

// Vulkan's name/description fields are bounded; produce a complete JSON string.
static void quoted(char out[VK_MAX_DESCRIPTION_SIZE * 6 + 3], const char *text) {
    char *p = out;
    *p++ = '"';
    for (unsigned i = 0; i < VK_MAX_DESCRIPTION_SIZE && text[i]; i++) {
        unsigned char c = (unsigned char)text[i];
        if (c == '"' || c == '\\') { *p++ = '\\'; *p++ = (char)c; }
        else if (c < 0x20 || c >= 0x7f) { snprintf(p, 7, "\\u%04x", c); p += 6; }
        else *p++ = (char)c;
    }
    *p++ = '"'; *p = 0;
}

void vk_pipeline_stats(VkPipeline pipeline, const char *kind, uint64_t id0, uint64_t id1,
                       uint64_t state_key, const uint32_t local[3]) {
    if (!g_vk.pipeline_stats || !pipeline) return;
    pthread_once(&once, init);
    if (!get_properties || !get_statistics) return;
    VkPipelineInfoKHR pi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR, .pipeline = pipeline};
    uint32_t count = 0;
    VkResult r = get_properties(g_vk.dev, &pi, &count, NULL);
    if (r != VK_SUCCESS || !count || count > 64) {
        LOG("[pipeline-exec] properties unavailable: result %d count %u\n", r, count);
        return;
    }
    VkPipelineExecutablePropertiesKHR *props = calloc(count, sizeof *props);
    if (!props) return;
    for (unsigned i = 0; i < count; i++) props[i].sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR;
    r = get_properties(g_vk.dev, &pi, &count, props);
    if (r != VK_SUCCESS) { LOG("[pipeline-exec] incomplete properties: %d\n", r); free(props); return; }
    for (unsigned i = 0; i < count; i++) {
        char name[VK_MAX_DESCRIPTION_SIZE * 6 + 3], desc[sizeof name];
        quoted(name, props[i].name); quoted(desc, props[i].description);
        TRACE_LOG("[pipeline-exec] {\"pipeline\":\"%016llx\",\"kind\":\"%s\",\"id0\":\"%016llx\",\"id1\":\"%016llx\","
                  "\"state_key\":\"%016llx\",\"index\":%u,\"stages\":%u,\"subgroup_size\":%u,\"name\":%s,\"description\":%s,"
                  "\"local\":[%u,%u,%u],\"gpl\":%d,\"gpl_lto\":%d}\n",
                  (unsigned long long)(uintptr_t)pipeline, kind, (unsigned long long)id0, (unsigned long long)id1,
                  (unsigned long long)state_key, i, props[i].stages, props[i].subgroupSize, name, desc,
                  local ? local[0] : 0, local ? local[1] : 0, local ? local[2] : 0, g_vk.gpl, g_vk.gpl_lto);
        VkPipelineExecutableInfoKHR ei = {.sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR,
                                          .pipeline = pipeline, .executableIndex = i};
        uint32_t ns = 0;
        r = get_statistics(g_vk.dev, &ei, &ns, NULL);
        if (r != VK_SUCCESS || !ns || ns > 256) {
            LOG("[pipeline-exec] statistics unavailable: result %d count %u\n", r, ns);
            continue;
        }
        VkPipelineExecutableStatisticKHR *stats = calloc(ns, sizeof *stats);
        if (!stats) continue;
        for (unsigned j = 0; j < ns; j++) stats[j].sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR;
        r = get_statistics(g_vk.dev, &ei, &ns, stats);
        if (r != VK_SUCCESS) { LOG("[pipeline-exec] incomplete statistics: %d\n", r); free(stats); continue; }
        for (unsigned j = 0; j < ns; j++) {
            char value[64];
            switch (stats[j].format) {
            case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR:
                snprintf(value, sizeof value, "%s", stats[j].value.b32 ? "true" : "false"); break;
            case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR:
                snprintf(value, sizeof value, "%" PRId64, stats[j].value.i64); break;
            case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR:
                snprintf(value, sizeof value, "%" PRIu64, stats[j].value.u64); break;
            case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_FLOAT64_KHR:
                if (isfinite(stats[j].value.f64)) snprintf(value, sizeof value, "%.17g", stats[j].value.f64);
                else snprintf(value, sizeof value, "null");
                break;
            default: snprintf(value, sizeof value, "null"); break;
            }
            quoted(name, stats[j].name); quoted(desc, stats[j].description);
            TRACE_LOG("[pipeline-stat] {\"pipeline\":\"%016llx\",\"index\":%u,\"name\":%s,\"description\":%s,\"format\":%d,\"value\":%s}\n",
                      (unsigned long long)(uintptr_t)pipeline, i, name, desc, stats[j].format, value);
        }
        free(stats);
    }
    free(props);
}
