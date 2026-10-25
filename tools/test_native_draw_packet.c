// Exercise the real producer/recorder queue with mock driver entry points. Verify copied payloads,
// sparse vertex binding runs, signed base vertices, query ordering, worker callbacks and ring wrap.
#include <assert.h>
#include "../runtime/nvn/vulkan/vkq.c"

#define DEFINE_VKD(n) PFN_vk##n vkd_##n;
VKD_FUNCS(DEFINE_VKD)
#undef DEFINE_VKD
void kernel_sample_host_thread(void) {}
void rt_fatal_trace(void) { abort(); }
void vk_devlost_dump(void) { abort(); }
void vk_timing_driver(uint32_t n, const VkSubmitInfo2 *si, bool done) { (void)n; (void)si; (void)done; }
bool vk_frame_trace_on;
uint64_t plat_now_ns(void) { return 0; }
uint64_t renderer_present_count(void) { return 0; }
int plat_trace_log(const char *format, ...) { (void)format; return 0; }
void vk_frame_trace_event(const char *name,unsigned lane,uint64_t frame,uint64_t seq,uint64_t begin,uint64_t end,unsigned draws,uint64_t shader) {
    (void)name;(void)lane;(void)frame;(void)seq;(void)begin;(void)end;(void)draws;(void)shader;assert(!vk_frame_trace_on);
}

static unsigned events[32], nevents;
static bool full;
static int worker_result;
static void event(unsigned id) { assert(nevents < 32); events[nevents++] = id; }
static unsigned native_pending;
void vk_native_batch_replay(void *p) { event(*(unsigned*)p);native_pending++; }
void vk_native_batch_flush(void) { if(native_pending){event(200+native_pending);native_pending=0;} }
static void ordinary_command(void *p) { event(*(unsigned*)p); }
#define HANDLE(T, n) ((T)(uintptr_t)(n))
static void VKAPI_CALL sets(VkCommandBuffer cb, VkPipelineBindPoint bp, VkPipelineLayout l, uint32_t first,
                           uint32_t n, const VkDescriptorSet *s, uint32_t nd, const uint32_t *d) {
    assert(cb == HANDLE(VkCommandBuffer, 1) && bp == VK_PIPELINE_BIND_POINT_GRAPHICS);
    assert(l == HANDLE(VkPipelineLayout, 2) && first == 0 && n == 2 && nd == 32);
    assert(s[0] == HANDLE(VkDescriptorSet, 3) && s[1] == HANDLE(VkDescriptorSet, 4));
    for (unsigned i = 0; i < nd; i++) assert(d[i] == 256 * (i + 1));
    event(1);
}
static void VKAPI_CALL index_buffer(VkCommandBuffer cb, VkBuffer b, VkDeviceSize off, VkIndexType ty) {
    assert(cb == HANDLE(VkCommandBuffer, 1) && b == HANDLE(VkBuffer, 7));
    assert(off == 96 && ty == VK_INDEX_TYPE_UINT32);
    event(2);
}
static void VKAPI_CALL vertices(VkCommandBuffer cb, uint32_t first, uint32_t n, const VkBuffer *b, const VkDeviceSize *off) {
    assert(cb == HANDLE(VkCommandBuffer, 1));
    assert((first == 0 && n == 2) || (first == 4 && n == 1) || (first == 16 && n == 1));
    for (unsigned i = 0; i < n; i++) {
        assert(b[i] == HANDLE(VkBuffer, 100 + first + i));
        assert(off[i] == 1000 + (first + i) * 8);
    }
    event(10 + first);
}
static void VKAPI_CALL begin_query(VkCommandBuffer cb, VkQueryPool p, uint32_t q, VkQueryControlFlags flags) {
    assert(cb == HANDLE(VkCommandBuffer, 1) && p == HANDLE(VkQueryPool, 88) && q == 41 && flags == VK_QUERY_CONTROL_PRECISE_BIT);
    event(3);
}
static void VKAPI_CALL end_query(VkCommandBuffer cb, VkQueryPool p, uint32_t q) {
    assert(cb == HANDLE(VkCommandBuffer, 1) && p == HANDLE(VkQueryPool, 88) && q == 41);
    event(6);
}
static void VKAPI_CALL indexed_draw(VkCommandBuffer cb, uint32_t n, uint32_t inst, uint32_t first, int32_t base, uint32_t bi) {
    assert(full && cb == HANDLE(VkCommandBuffer, 1) && n == 6 && inst == 2 && first == 5 && base == -3 && bi == 9);
    event(4);
}
static void VKAPI_CALL draw(VkCommandBuffer cb, uint32_t n, uint32_t inst, uint32_t first, uint32_t bi) {
    assert(!full && cb == HANDLE(VkCommandBuffer, 1) && n == 3 && inst == 1 && first == 0 && bi == 6);
    assert(worker_result == 7);
    event(5);
}
static void job(VkCommandBuffer cb, VkPipelineLayout l, void *arg) {
    assert(cb == HANDLE(VkCommandBuffer, 1) && l == HANDLE(VkPipelineLayout, 2) && arg == &worker_result);
    worker_result = 7;
    event(8);
}
static VkqDraw packet(void) {
    VkqDraw d = {.cb = HANDLE(VkCommandBuffer, 1), .layout = HANDLE(VkPipelineLayout, 2),
        .sets = {HANDLE(VkDescriptorSet, 3), HANDLE(VkDescriptorSet, 4)}, .ndyn = 32,
        .vertex_mask = 3 | (1u << 4) | (1u << 16), .index = HANDLE(VkBuffer, 7), .index_offset = 96,
        .index_type = VK_INDEX_TYPE_UINT32, .bind_sets = true, .bind_index = true, .indexed = true,
        .count = 6, .instances = 2, .first = 5, .base_vertex = -3, .base_instance = 9,
        .query_pool = HANDLE(VkQueryPool, 88), .query = 41, .query_flags = VK_QUERY_CONTROL_PRECISE_BIT};
    for (unsigned i = 0; i < 32; i++) d.dyn[i] = 256 * (i + 1);
    for (unsigned i = 0; i < 17; i++) { d.vertex[i] = HANDLE(VkBuffer, 100 + i); d.offsets[i] = 1000 + i * 8; }
    return d;
}
static void check(const unsigned *expected, unsigned n) {
    assert(nevents == n && !memcmp(events, expected, n * sizeof *expected));
    nevents = 0;
}
typedef struct { uint64_t serial, inverse; uint8_t data[48]; } StressRecord;
static uint64_t stress_seen;
static void stress_replay(void *p) {
    const StressRecord *r = p;
    assert(r->serial == stress_seen && r->inverse == ~r->serial);
    for (unsigned i = 0; i < sizeof r->data; i++) assert(r->data[i] == (uint8_t)(r->serial + i));
    if ((stress_seen & 1023) == 0) { struct timespec delay = {0, 100000}; nanosleep(&delay, NULL); }
    stress_seen++;
}
int main(void) {
    vkq_native_multi_draw=true; // initialized before recorder creation
    vkd_CmdBindDescriptorSets = sets;
    vkd_CmdBindIndexBuffer = index_buffer;
    vkd_CmdBindVertexBuffers = vertices;
    vkd_CmdBeginQuery = begin_query;
    vkd_CmdEndQuery = end_query;
    vkd_CmdDrawIndexed = indexed_draw;
    vkd_CmdDraw = draw;
    const unsigned expected[] = {1, 2, 10, 14, 26, 3, 4, 6};
    full = true;
    VkqDraw d = packet();
    vkq_draw_packet(&d);
    check(expected, 8);
    // Start at the end of the ring: the next packet must insert and consume wrap padding.
    wlocal = rcache = RING_SIZE - 32;
    atomic_store(&rpos, wlocal);
    atomic_store(&wpos, wlocal);
    vkq_defer_begin();
    assert(vkq_on);
    vkq_draw_packet(&d);
    memset(&d, 0xcc, sizeof d);  // caller may immediately reuse its argument storage
    vkq_defer_end();
    assert(wlocal > RING_SIZE && atomic_load(&rpos) == wlocal);
    check(expected, 8);
    full = false;
    d = (VkqDraw){.cb = HANDLE(VkCommandBuffer, 1), .layout = HANDLE(VkPipelineLayout, 2),
        .bind_job = job, .job = &worker_result, .count = 3, .instances = 1, .base_instance = 6};
    vkq_defer_begin();
    vkq_draw_packet(&d);
    memset(&d, 0, sizeof d);
    vkq_defer_end();
    const unsigned worker_expected[] = {8, 5};
    check(worker_expected, 2);
    // Repeated idle/wake/drain transitions, delayed consumer, and immediate
    // caller reuse. This runs against both pthread and native address waits.
    for (unsigned batch = 0; batch < 2048; batch++) {
        vkq_defer_begin();
        if ((batch & 31) == 0) { struct timespec delay = {0, 1000000}; nanosleep(&delay, NULL); }
        for (unsigned j = 0; j < 64; j++) {
            StressRecord r = {.serial = (uint64_t)batch * 64 + j};
            r.inverse = ~r.serial;
            for (unsigned i = 0; i < sizeof r.data; i++) r.data[i] = (uint8_t)(r.serial + i);
            assert(vkq_custom(stress_replay, &r, sizeof r));
            memset(&r, 0xcc, sizeof r);
        }
        vkq_defer_end();
        assert(stress_seen == (uint64_t)(batch + 1) * 64);
    }
    if (getenv("SAVAGEWIND_VK_ADDRESS_WAIT") && atoi(getenv("SAVAGEWIND_VK_ADDRESS_WAIT"))) assert(vkq_address_wait);
    puts("native draw packet: direct, deferred, wrap, payload lifetime and ordering passed");
    puts("recorder wait: 131072 ordered records across 2048 idle/drain transitions passed");
    vkq_defer_begin();
    unsigned value=100;assert(vkq_custom(vk_native_batch_replay,&value,sizeof value));
    value=101;assert(vkq_custom(vk_native_batch_replay,&value,sizeof value));
    value=103;assert(vkq_custom(ordinary_command,&value,sizeof value));
    value=104;assert(vkq_custom(vk_native_batch_replay,&value,sizeof value));
    vkq_defer_end();
    const unsigned flush_expected[]={100,101,202,103,104,201};check(flush_expected,6);
    puts("native batch ordering: grouped callbacks flush before ordinary commands and drains");
    return 0;
}
