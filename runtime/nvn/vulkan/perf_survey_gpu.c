#include "vk.h"
#include "../../perf_survey.h"

// Private diagnostic dispatch, appended after the real frame. Its result is observable storage,
// so the dependent integer loop cannot be optimized away. No game resource is modified.
#define SURVEY_GPU_SLOTS 64
static VkPipeline pipe;
static VkPipelineLayout layout;
static VkDescriptorSetLayout set_layout;
static VkQueryPool queries;
static uint64_t timestamp_mask;
static uint32_t iterations = 20000;
static struct { uint64_t seq, frame; uint32_t iterations; int block; } pending[SURVEY_GPU_SLOTS];

static void survey_gpu_init(void) {
    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(g_vk.phys, &nq, NULL);
    VkQueueFamilyProperties *q = calloc(nq, sizeof *q);
    vkGetPhysicalDeviceQueueFamilyProperties(g_vk.phys, &nq, q);
    unsigned bits = q[g_vk.qfam].timestampValidBits; free(q);
    if (!bits) FATAL("Survey GPU queue has no timestamp support\n");
    timestamp_mask = bits >= 64 ? UINT64_MAX : (1ull << bits) - 1;
    VkDescriptorSetLayoutBinding b = {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL};
    VkDescriptorSetLayoutCreateInfo si = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 1, .pBindings = &b};
    VK_CHECK(vkCreateDescriptorSetLayout(g_vk.dev, &si, NULL, &set_layout));
    VkPushConstantRange pc = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 8};
    VkPipelineLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1, .pSetLayouts = &set_layout, .pushConstantRangeCount = 1, .pPushConstantRanges = &pc};
    VK_CHECK(vkCreatePipelineLayout(g_vk.dev, &li, NULL, &layout));
    VkShaderModule shader = vk_compile_glsl(
        "#version 450\nlayout(local_size_x=32) in;\n"
        "layout(set=0,binding=0,std430) buffer Out { uint value[]; } result;\n"
        "layout(push_constant) uniform Push { uint count; uint seed; } pc;\n"
        "void main(){ uint x=pc.seed+gl_LocalInvocationIndex;"
        "for(uint i=0;i<pc.count;i++){x ^= x << 13; x ^= x >> 17; x ^= x << 5;}"
        "result.value[gl_LocalInvocationIndex]=x;}\n", 5, "diagnostic-survey-delay");
    if (!shader) FATAL("Survey GPU shader compilation failed\n");
    VkComputePipelineCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                  .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = shader, .pName = "main"}, .layout = layout};
    VK_CHECK(vkCreateComputePipelines(g_vk.dev, VK_NULL_HANDLE, 1, &ci, NULL, &pipe));
    vkDestroyShaderModule(g_vk.dev, shader, NULL);
    VkQueryPoolCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
        .queryType = VK_QUERY_TYPE_TIMESTAMP, .queryCount = 2 * SURVEY_GPU_SLOTS};
    VK_CHECK(vkCreateQueryPool(g_vk.dev, &qi, NULL, &queries));
}

void vk_perf_survey_gpu(VkCommandBuffer cb) {
    if (perf_survey_mode != SURVEY_GPU) return;
    uint64_t elapsed = plat_now_ns() - perf_survey_epoch;
    if (elapsed < 20000000000ull) return;
    if (!pipe) survey_gpu_init();
    bool calibrate = elapsed < perf_survey_start_ns - 5000000000ull;
    uint64_t completed = vk_completed_seq();
    for (unsigned s = 0; s < SURVEY_GPU_SLOTS; s++) {
        if (!pending[s].seq || pending[s].seq > completed) continue;
        uint64_t ticks[2];
        VkResult r = vkGetQueryPoolResults(g_vk.dev, queries, 2*s, 2, sizeof ticks, ticks, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
        if (r == VK_NOT_READY) continue;
        VK_CHECK(r);
        double ns = ((ticks[1] - ticks[0]) & timestamp_mask) * g_vk.props.limits.timestampPeriod;
        if (calibrate && ns > 1000 && pending[s].iterations == iterations) {
            double next = iterations * 500000.0 / ns;
            if (next < 100) next = 100; if (next > 2000000) next = 2000000;
            iterations = (uint32_t)(0.75 * iterations + 0.25 * next);
        }
        TRACE_LOG("[survey-gpu-dose] frame %llu block %d ns %.0f iterations %u\n",
            (unsigned long long)pending[s].frame, pending[s].block, ns, pending[s].iterations);
        pending[s].seq = 0;
    }
    if (!calibrate && !perf_survey_on(SURVEY_GPU)) return;
    uint64_t frame = atomic_load(&perf_survey_frame);
    unsigned s = frame % SURVEY_GPU_SLOTS;
    if (pending[s].seq) FATAL("Survey GPU query ring exhausted\n");
    VkSpan out = vk_transient(128, 256);
    VkDescriptorSet set = vk_alloc_set(set_layout);
    VkDescriptorBufferInfo bi = {out.buf, out.off, 128};
    VkWriteDescriptorSet wi = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set,
        .dstBinding = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &bi};
    vkUpdateDescriptorSets(g_vk.dev, 1, &wi, 0, NULL);
    vk_full_barrier(cb);
    vkCmdResetQueryPool(cb, queries, 2*s, 2);
    vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, queries, 2*s);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, NULL);
    uint32_t pc[2] = {iterations, (uint32_t)frame};
    vkCmdPushConstants(cb, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, pc);
    vkCmdDispatch(cb, 1, 1, 1);
    vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, queries, 2*s + 1);
    vk_full_barrier(cb);
    vk_dyn_forget();
    pending[s].seq = vk_pending_seq(); pending[s].frame = frame;
    pending[s].iterations = iterations; pending[s].block = atomic_load(&perf_survey_block);
}
