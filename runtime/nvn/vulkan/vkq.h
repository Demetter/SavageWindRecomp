// Command recording on a thread of its own (vkq.c). While a list is translated (backend_execute), the submission
// thread doesn't call the driver: each Vulkan command, with what its pointers point to, goes into a ring that the
// recording thread replays into the command buffers (and submits, in order). Most of the translation's time was
// inside the driver's vkCmd* functions; now the two halves run side by side and a list takes about the longer of
// the two instead of their sum. Only list translation defers (vkq_defer_begin / vkq_defer_end, which waits for the
// recording to catch up): the present, the overlay, uploads outside lists and every wait on the GPU record directly,
// as before. SAVAGEWIND_VK_NO_DEFER=1: no recording thread.
#pragma once
#include <stdbool.h>
#include <vulkan/vulkan.h>

extern __thread bool vkq_on;  // this thread's Vulkan commands go to the recording thread
extern bool vkq_native_multi_draw; // device initialization sets this before recorder creation
void vkq_defer_begin(void);   // (the submission thread, at the start of a list)
void vkq_defer_end(void);     // all of it recorded and submitted: direct calls are safe again
void vkq_drain(void);         // everything deferred so far recorded and submitted (deferring goes on)
void vkq_trace_frame(uint64_t frame); // diagnostic wait deltas, separate from descriptor-job waits
extern bool vkq_address_wait; // native Windows waits, selected before recorder/descriptor workers start
void vkq_wait_address(volatile void *address, uint64_t expected, size_t bytes);
void vkq_wake_address(void *address, bool all);
// fn(arg) on the recording thread, after the commands deferred before it (size bytes of arg copied); false: not
// deferring (the caller runs it itself)
bool vkq_custom(void (*fn)(void *), const void *arg, size_t size);
// Diagnostic only. Starts tracking producer-side ordering boundaries, excluding
// draw/geometry/descriptor bindings and the supplied descriptor-job callback.
// All other commands (including queries, barriers and submissions) advance it.
uint64_t vkq_batch_probe_token(void (*descriptor_bind)(void *));

// Prepared native draw. The queue stores one compact packet rather than a
// record for every descriptor/index/vertex bind and draw. Guest state and
// memory dependencies have already been resolved by the producer.
typedef struct {
    VkCommandBuffer cb;
    VkPipelineLayout layout;
    VkDescriptorSet sets[2];
    uint32_t dyn[32], ndyn;
    void (*bind_job)(VkCommandBuffer, VkPipelineLayout, void *);
    void *job;
    uint32_t vertex_mask;
    VkBuffer vertex[17];
    VkDeviceSize offsets[17];
    VkBuffer index;
    VkDeviceSize index_offset;
    VkIndexType index_type;
    bool bind_sets, bind_index, indexed;
    uint32_t count, instances, first, base_instance;
    int32_t base_vertex;
    VkQueryPool query_pool;
    uint32_t query;
    VkQueryControlFlags query_flags;
} VkqDraw;
void vkq_draw_packet(const VkqDraw *draw);

// The deferring versions of vkd.h's functions (the driver's own when vkq_on is false)
VKAPI_ATTR VkResult VKAPI_CALL vkq_BeginCommandBuffer(VkCommandBuffer cb, const VkCommandBufferBeginInfo *bi);
VKAPI_ATTR VkResult VKAPI_CALL vkq_EndCommandBuffer(VkCommandBuffer cb);
VKAPI_ATTR VkResult VKAPI_CALL vkq_QueueSubmit2(VkQueue q, uint32_t n, const VkSubmitInfo2 *si, VkFence f);
VKAPI_ATTR VkResult VKAPI_CALL vkq_WaitSemaphores(VkDevice d, const VkSemaphoreWaitInfo *wi, uint64_t timeout);
VKAPI_ATTR void VKAPI_CALL vkq_CmdBeginQuery(VkCommandBuffer cb, VkQueryPool qp, uint32_t q, VkQueryControlFlags f);
VKAPI_ATTR void VKAPI_CALL vkq_CmdBeginRendering(VkCommandBuffer cb, const VkRenderingInfo *ri);
VKAPI_ATTR void VKAPI_CALL vkq_CmdBindDescriptorSets(VkCommandBuffer cb, VkPipelineBindPoint bp, VkPipelineLayout l,
                                                     uint32_t first, uint32_t n, const VkDescriptorSet *sets,
                                                     uint32_t ndyn, const uint32_t *dyn);
VKAPI_ATTR void VKAPI_CALL vkq_CmdBindIndexBuffer(VkCommandBuffer cb, VkBuffer b, VkDeviceSize off, VkIndexType t);
VKAPI_ATTR void VKAPI_CALL vkq_CmdBindPipeline(VkCommandBuffer cb, VkPipelineBindPoint bp, VkPipeline p);
VKAPI_ATTR void VKAPI_CALL vkq_CmdBindVertexBuffers(VkCommandBuffer cb, uint32_t first, uint32_t n, const VkBuffer *b,
                                                    const VkDeviceSize *off);
VKAPI_ATTR void VKAPI_CALL vkq_CmdBlitImage(VkCommandBuffer cb, VkImage s, VkImageLayout sl, VkImage d, VkImageLayout dl,
                                            uint32_t n, const VkImageBlit *r, VkFilter f);
VKAPI_ATTR void VKAPI_CALL vkq_CmdClearAttachments(VkCommandBuffer cb, uint32_t n, const VkClearAttachment *a,
                                                   uint32_t nr, const VkClearRect *r);
VKAPI_ATTR void VKAPI_CALL vkq_CmdClearColorImage(VkCommandBuffer cb, VkImage i, VkImageLayout l,
                                                  const VkClearColorValue *c, uint32_t n,
                                                  const VkImageSubresourceRange *r);
VKAPI_ATTR void VKAPI_CALL vkq_CmdClearDepthStencilImage(VkCommandBuffer cb, VkImage i, VkImageLayout l,
                                                         const VkClearDepthStencilValue *v, uint32_t n,
                                                         const VkImageSubresourceRange *r);
VKAPI_ATTR void VKAPI_CALL vkq_CmdCopyBuffer(VkCommandBuffer cb, VkBuffer s, VkBuffer d, uint32_t n,
                                             const VkBufferCopy *r);
VKAPI_ATTR void VKAPI_CALL vkq_CmdCopyBufferToImage(VkCommandBuffer cb, VkBuffer s, VkImage d, VkImageLayout dl,
                                                    uint32_t n, const VkBufferImageCopy *r);
VKAPI_ATTR void VKAPI_CALL vkq_CmdCopyImage(VkCommandBuffer cb, VkImage s, VkImageLayout sl, VkImage d, VkImageLayout dl,
                                            uint32_t n, const VkImageCopy *r);
VKAPI_ATTR void VKAPI_CALL vkq_CmdCopyImageToBuffer(VkCommandBuffer cb, VkImage s, VkImageLayout sl, VkBuffer d,
                                                    uint32_t n, const VkBufferImageCopy *r);
VKAPI_ATTR void VKAPI_CALL vkq_CmdDispatch(VkCommandBuffer cb, uint32_t x, uint32_t y, uint32_t z);
VKAPI_ATTR void VKAPI_CALL vkq_CmdDraw(VkCommandBuffer cb, uint32_t vc, uint32_t ic, uint32_t fv, uint32_t fi);
VKAPI_ATTR void VKAPI_CALL vkq_CmdDrawIndexed(VkCommandBuffer cb, uint32_t ic, uint32_t inst, uint32_t fi, int32_t vo,
                                              uint32_t finst);
VKAPI_ATTR void VKAPI_CALL vkq_CmdEndQuery(VkCommandBuffer cb, VkQueryPool qp, uint32_t q);
VKAPI_ATTR void VKAPI_CALL vkq_CmdEndRendering(VkCommandBuffer cb);
VKAPI_ATTR void VKAPI_CALL vkq_CmdPipelineBarrier2(VkCommandBuffer cb, const VkDependencyInfo *di);
VKAPI_ATTR void VKAPI_CALL vkq_CmdPushConstants(VkCommandBuffer cb, VkPipelineLayout l, VkShaderStageFlags st,
                                                uint32_t off, uint32_t size, const void *data);
VKAPI_ATTR void VKAPI_CALL vkq_CmdResetQueryPool(VkCommandBuffer cb, VkQueryPool qp, uint32_t first, uint32_t n);
VKAPI_ATTR void VKAPI_CALL vkq_CmdResolveImage(VkCommandBuffer cb, VkImage s, VkImageLayout sl, VkImage d,
                                               VkImageLayout dl, uint32_t n, const VkImageResolve *r);
VKAPI_ATTR void VKAPI_CALL vkq_CmdSetBlendConstants(VkCommandBuffer cb, const float c[4]);
VKAPI_ATTR void VKAPI_CALL vkq_CmdSetCullMode(VkCommandBuffer cb, VkCullModeFlags m);
VKAPI_ATTR void VKAPI_CALL vkq_CmdSetDepthBias(VkCommandBuffer cb, float c, float cl, float s);
VKAPI_ATTR void VKAPI_CALL vkq_CmdSetDepthBiasEnable(VkCommandBuffer cb, VkBool32 e);
VKAPI_ATTR void VKAPI_CALL vkq_CmdSetDepthCompareOp(VkCommandBuffer cb, VkCompareOp op);
VKAPI_ATTR void VKAPI_CALL vkq_CmdSetDepthTestEnable(VkCommandBuffer cb, VkBool32 e);
VKAPI_ATTR void VKAPI_CALL vkq_CmdSetDepthWriteEnable(VkCommandBuffer cb, VkBool32 e);
VKAPI_ATTR void VKAPI_CALL vkq_CmdSetFrontFace(VkCommandBuffer cb, VkFrontFace f);
VKAPI_ATTR void VKAPI_CALL vkq_CmdSetLineWidth(VkCommandBuffer cb, float w);
VKAPI_ATTR void VKAPI_CALL vkq_CmdSetPrimitiveRestartEnable(VkCommandBuffer cb, VkBool32 e);
VKAPI_ATTR void VKAPI_CALL vkq_CmdSetPrimitiveTopology(VkCommandBuffer cb, VkPrimitiveTopology t);
VKAPI_ATTR void VKAPI_CALL vkq_CmdSetScissor(VkCommandBuffer cb, uint32_t first, uint32_t n, const VkRect2D *r);
VKAPI_ATTR void VKAPI_CALL vkq_CmdSetStencilCompareMask(VkCommandBuffer cb, VkStencilFaceFlags f, uint32_t m);
VKAPI_ATTR void VKAPI_CALL vkq_CmdSetStencilOp(VkCommandBuffer cb, VkStencilFaceFlags f, VkStencilOp fail,
                                               VkStencilOp pass, VkStencilOp dfail, VkCompareOp op);
VKAPI_ATTR void VKAPI_CALL vkq_CmdSetStencilReference(VkCommandBuffer cb, VkStencilFaceFlags f, uint32_t r);
VKAPI_ATTR void VKAPI_CALL vkq_CmdSetStencilTestEnable(VkCommandBuffer cb, VkBool32 e);
VKAPI_ATTR void VKAPI_CALL vkq_CmdSetStencilWriteMask(VkCommandBuffer cb, VkStencilFaceFlags f, uint32_t m);
VKAPI_ATTR void VKAPI_CALL vkq_CmdSetViewport(VkCommandBuffer cb, uint32_t first, uint32_t n, const VkViewport *v);
VKAPI_ATTR void VKAPI_CALL vkq_CmdWriteTimestamp2(VkCommandBuffer cb, VkPipelineStageFlags2 st, VkQueryPool qp,
                                                  uint32_t q);
