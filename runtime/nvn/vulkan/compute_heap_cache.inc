// Opt-in working-set prototype for the measured expensive compute shader. No
// general heap replacement: uncovered pages fall back, and stores write through.
// A scratch buffer is reused only after a GPU execution dependency. Kept outside
// the geometry mirror so compute cannot invalidate cached vertex/index contents.
#define HC_PAGE_SHIFT 16
#define HC_PAGES ((unsigned)(HEAP_MAX >> HC_PAGE_SHIFT))
#define HC_SLOTS 1024u
static VkBuffer hc_buffer;
static VmaAllocation hc_memory;
static VkDescriptorBufferInfo hc_map;
static VkSpan hc_map_span;
static bool hc_audit;
static pthread_once_t hc_once = PTHREAD_ONCE_INIT;
static int hc_mode;
static void hc_init(void) { const char *e = getenv("SAVAGEWIND_VK_COMPUTE_CACHE"); hc_mode = e ? atoi(e) : 0; }
static int hc_on(void) { pthread_once(&hc_once, hc_init); return hc_mode; }

static void hc_prepare(VkCommandBuffer cb) {
    if (!hc_buffer) {
        VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = (uint64_t)HC_SLOTS << HC_PAGE_SHIFT,
            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        VK_CHECK(vmaCreateBuffer(g_vk.vma, &bi, &ai, &hc_buffer, &hc_memory, NULL));
        LOG("[compute-cache] 64 MiB device-local working set, write-through guest heap; audit %d\n", hc_on() == 2);
    }
    static uint32_t map[HC_PAGES + 4];
    static VkBufferCopy copies[HC_SLOTS];
    memset(map, 0, sizeof map);
    unsigned slots = 0, ncopy = 0;
    // Bound resources describe the likely working set. Correctness does not rely
    // on complete coverage: a shader pointer outside this set still reads gmem.
    for (int kind = 0; kind < 2; kind++) for (int i = 0; i < 16; i++) {
        Range r = kind ? st.sbuf[5][i] : st.ubuf[5][i];
        if (!gmem_buf || !r.size || r.addr < HEAP_BASE || r.size > (8u << 20) ||
            r.addr - HEAP_BASE >= gmem_size || r.size > gmem_size - (r.addr - HEAP_BASE)) continue;
        uint64_t first = (r.addr - HEAP_BASE) >> HC_PAGE_SHIFT;
        uint64_t last = (r.addr - HEAP_BASE + r.size - 1) >> HC_PAGE_SHIFT;
        for (uint64_t p = first; p <= last && slots < HC_SLOTS; p++) {
            if (map[p] || ((p + 1) << HC_PAGE_SHIFT) > gmem_size) continue;
            uint64_t src = p << HC_PAGE_SHIFT, dst = (uint64_t)slots << HC_PAGE_SHIFT;
            map[p] = ++slots;
            if (ncopy && copies[ncopy-1].srcOffset + copies[ncopy-1].size == src &&
                copies[ncopy-1].dstOffset + copies[ncopy-1].size == dst) copies[ncopy-1].size += 1u << HC_PAGE_SHIFT;
            else copies[ncopy++] = (VkBufferCopy){src, dst, 1u << HC_PAGE_SHIFT};
        }
    }
    uint64_t frame = renderer_present_count();
    // Menu duration varies with pipeline prewarming. Start after 64 actual target
    // dispatches, not at an absolute present that may still be a loading screen.
    static unsigned dispatches;
    static uint64_t audit_start = UINT64_MAX;
    if (hc_on() == 2 && ++dispatches >= 64 && audit_start == UINT64_MAX) audit_start = frame;
    hc_audit = audit_start != UINT64_MAX && frame - audit_start < 16;
    map[HC_PAGES + 2] = hc_audit;
    hc_map_span = vk_transient_copy(map, sizeof map, 256);
    hc_map = (VkDescriptorBufferInfo){hc_map_span.buf, hc_map_span.off, sizeof map};
    if (ncopy) {
        VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
            .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT};
        VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
        vkCmdPipelineBarrier2(cb, &di);
        vkCmdCopyBuffer(cb, gmem_buf, hc_buffer, ncopy, copies);
        mb.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        mb.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        mb.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        mb.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        vkCmdPipelineBarrier2(cb, &di);
    }
    if (hc_audit) LOG("[compute-cache] audit frame %llu: %u pages / %u copy regions\n", (unsigned long long)frame, slots, ncopy);
}

static void hc_check(VkCommandBuffer cb) {
    if (!hc_audit) return;
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, .srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT, .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
    vkCmdPipelineBarrier2(cb, &di);
    vk_submit_extra(VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE);
    vk_wait_seq(submit_seq);
    for (int i = 0; i < atomic_load(&nchunks); i++) if (chunks[i].buf == hc_map.buffer) {
        VK_CHECK(vmaInvalidateAllocation(g_vk.vma, chunks[i].mem, hc_map.offset + HC_PAGES * 4ull, 16));
        break;
    }
    const uint32_t *map = hc_map_span.ptr;
    LOG("[compute-cache] checked %u cached words / %u total reads; mismatches %u\n", map[HC_PAGES+1], map[HC_PAGES+3], map[HC_PAGES]);
    if (map[HC_PAGES]) FATAL("compute working-set reads differ from guest heap\n");
}
