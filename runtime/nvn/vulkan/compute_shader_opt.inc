// Diagnostic only. The original game dispatch still executes normally after
// two store-journal variants compare exact outputs using the same inputs.
// Results describe this shader/input coverage, not all shaders or all scenes.
static void co_audit(CsProg *cp, uint32_t gx, uint32_t gy, uint32_t gz) {
    static unsigned dispatches, checks;
    static const ShaderInfo *si;
    static uint32_t local[3];
    static Prog p;
    static VkPipeline pipelines[2];
    bool split_groups = co_on() == 4;
    if (split_groups && (cp->local[0] != 32 || cp->local[1] != 1 || cp->local[2] != 1 ||
                        gx > g_vk.props.limits.maxComputeWorkGroupCount[0] / 4))
        FATAL("compute group audit cannot preserve these dispatch dimensions\n");
    ++dispatches;
    // Cover both scene population and the later full workload. An early fixed
    // window alone can finish before the expensive 16-group dispatch appears.
    if (dispatches < 64 || checks >= 96 || (checks >= 48 && dispatches < 1024)) return;
    uint64_t invocations = 1;
    uint32_t groups[3] = {gx, gy, gz};
    for (unsigned i = 0; i < 3; i++) {
        if (!groups[i] || invocations > 65536u / groups[i])
            FATAL("compute optimizer audit exceeds invocation capacity\n");
        invocations *= groups[i];
        if (!cp->local[i] || invocations > 65536u / cp->local[i])
            FATAL("compute optimizer audit exceeds invocation capacity\n");
        invocations *= cp->local[i];
    }
    if (!si || si != cp->si || memcmp(local, cp->local, sizeof local)) {
        // Each previous audit waited for its own submission before returning;
        // its pipelines/layout are no longer in flight here.
        for (unsigned i = 0; i < 2; i++) if (pipelines[i]) vkDestroyPipeline(g_vk.dev, pipelines[i], NULL);
        if (p.layout) vkDestroyPipelineLayout(g_vk.dev, p.layout, NULL);
        if (p.set_layout[0]) vkDestroyDescriptorSetLayout(g_vk.dev, p.set_layout[0], NULL);
        memset(&p, 0, sizeof p); memset(pipelines, 0, sizeof pipelines);
        for (unsigned variant = 0; variant < 2; variant++) {
            char *split = variant && split_groups ? shader_compute_groups(cp->si->msl) : NULL;
            if (variant && split_groups && !split) FATAL("compute group audit source rejected\n");
            char *source = shader_store_audit(split ? split : cp->si->msl);
            free(split);
            if (!source) FATAL("compute optimizer audit cannot safely redirect this shader's stores\n");
            size_t words = 0;
            uint32_t *spv = vk_glsl_to_spirv_mode(source, STAGE_COMPUTE, "compute-store-audit", &words, variant != 0);
            free(source);
            if (!spv) FATAL("compute optimizer audit shader compilation failed\n");
            VkBindingInfo binds[MAX_BIND];
            int n = vk_reflect(spv, words, binds, MAX_BIND);
            if (!variant) {
                p.nbinds = n; memcpy(p.binds, binds, n * sizeof binds[0]);
                p.layout = make_layout(p.binds, n, p.set_layout, 1, VK_SHADER_STAGE_COMPUTE_BIT, 0);
            } else {
                // Optimizing away an unused declaration is legal; the common
                // layout may contain extra bindings. Every remaining use must
                // still match the original layout and descriptor type.
                for (int i = 0; i < n; i++) {
                    bool found = false;
                    for (int j = 0; j < p.nbinds; j++)
                        found |= binds[i].set == p.binds[j].set && binds[i].binding == p.binds[j].binding && binds[i].type == p.binds[j].type;
                    if (!found) FATAL("compute optimizer audit descriptor layout changed\n");
                }
            }
            VkShaderModule module = vk_module(spv, words); free(spv);
            VkSpecializationMapEntry entries[3] = {{0,0,4},{1,4,4},{2,8,4}};
            uint32_t physical_local[3] = {variant && split_groups ? 8 : cp->local[0], cp->local[1], cp->local[2]};
            VkSpecializationInfo spec = {3, entries, sizeof physical_local, physical_local};
            VkComputePipelineCreateInfo info = {.sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                .layout=p.layout, .stage={.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage=VK_SHADER_STAGE_COMPUTE_BIT, .module=module, .pName="main", .pSpecializationInfo=&spec}};
            VK_CHECK(vkCreateComputePipelines(g_vk.dev, g_vk.pcache, 1, &info, NULL, &pipelines[variant]));
            vkDestroyShaderModule(g_vk.dev, module, NULL);
        }
        si = cp->si; memcpy(local, cp->local, sizeof local);
    }
    uint64_t bytes = invocations * STORE_AUDIT_WORDS * sizeof(uint32_t);
    VkSpan results[2] = {vk_transient(bytes, 256), vk_transient(bytes, 256)};
    VkCommandBuffer cb = vk_cmd();
    for (unsigned variant = 0; variant < 2; variant++) {
        memset(results[variant].ptr, 0, (size_t)bytes);
        compute_audit_output = (VkDescriptorBufferInfo){results[variant].buf, results[variant].off, bytes};
        VkDescriptorSet set = stage_set(&p, p.set_layout[0], 0, 5, cp->si, NULL, NULL, gx, gy, gz, NULL, NULL, NULL);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines[variant]);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p.layout, 0, 1, &set, 0, NULL);
        vkCmdDispatch(cb, variant && split_groups ? gx * 4 : gx, gy, gz);
    }
    VkMemoryBarrier2 barrier = {.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .srcStageMask=VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, .srcAccessMask=VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        .dstStageMask=VK_PIPELINE_STAGE_2_HOST_BIT, .dstAccessMask=VK_ACCESS_2_HOST_READ_BIT};
    VkDependencyInfo dependency = {.sType=VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount=1, .pMemoryBarriers=&barrier};
    vkCmdPipelineBarrier2(cb, &dependency);
    vk_submit_extra(VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE);
    vk_wait_seq(submit_seq);
    for (unsigned variant = 0; variant < 2; variant++) {
        for (int i = 0; i < atomic_load(&nchunks); i++) if (chunks[i].buf == results[variant].buf) {
            VK_CHECK(vmaInvalidateAllocation(g_vk.vma, chunks[i].mem, results[variant].off, bytes)); break;
        }
    }
    const uint32_t *a = results[0].ptr, *b = results[1].ptr;
    StoreAuditStats stats;
    int result = store_audit_check(a, b, (size_t)invocations, &stats);
    LOG("[compute-opt-audit] frame %llu groups %ux%ux%u invocations %llu stores %llu reads %llu differing words %llu result %d\n",
        (unsigned long long)renderer_present_count(), gx, gy, gz, (unsigned long long)invocations,
        (unsigned long long)stats.stores, (unsigned long long)stats.reads, (unsigned long long)stats.differences, result);
    if (result < 0) FATAL("compute optimizer audit invalid: %d (overflow/coverage/allocation/inter-invocation dependency)\n", result);
    if (result) FATAL("optimized compute outputs differ from original\n");
    checks++;
}
