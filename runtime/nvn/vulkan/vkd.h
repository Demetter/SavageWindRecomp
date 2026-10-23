// The submission thread's Vulkan calls, straight to the driver. Called through vulkan-1.dll, each one went through
// the loader's trampoline (its dispatch table looked up from the handle) first: ~10% of the submission thread in
// BOTW's open world, at ~20 calls a draw and ~5,000 draws a frame. vkd_load (after vkCreateDevice) takes each
// function's device-level entry point from vkGetDeviceProcAddr, and vk.h's names below call those; a function the
// device doesn't give keeps the loader's. vkd.c, which defines them, sees the loader's names (VKD_NO_REDIRECT).
// The recording calls go through vkq.h's versions, which defer them to the recording thread when it's on.
#pragma once
#include <vulkan/vulkan.h>

#define VKD_FUNCS(X)                                                                                                  \
    X(AllocateDescriptorSets) X(BeginCommandBuffer) X(CmdBeginQuery) X(CmdBeginRendering) X(CmdBindDescriptorSets)    \
    X(CmdBindIndexBuffer) X(CmdBindPipeline) X(CmdBindVertexBuffers) X(CmdBlitImage) X(CmdClearAttachments)           \
    X(CmdClearColorImage) X(CmdClearDepthStencilImage) X(CmdCopyBuffer) X(CmdCopyBufferToImage) X(CmdCopyImage)       \
    X(CmdCopyImageToBuffer) X(CmdDispatch) X(CmdDraw) X(CmdDrawIndexed) X(CmdEndQuery) X(CmdEndRendering)             \
    X(CmdPipelineBarrier2) X(CmdPushConstants) X(CmdResetQueryPool) X(CmdResolveImage) X(CmdSetBlendConstants)        \
    X(CmdSetCullMode) X(CmdSetDepthBias) X(CmdSetDepthBiasEnable) X(CmdSetDepthCompareOp) X(CmdSetDepthTestEnable)    \
    X(CmdSetDepthWriteEnable) X(CmdSetFrontFace) X(CmdSetLineWidth) X(CmdSetPrimitiveRestartEnable)                   \
    X(CmdSetPrimitiveTopology) X(CmdSetScissor) X(CmdSetStencilCompareMask) X(CmdSetStencilOp)                        \
    X(CmdSetStencilReference) X(CmdSetStencilTestEnable) X(CmdSetStencilWriteMask) X(CmdSetViewport)                  \
    X(CmdWriteTimestamp2) X(EndCommandBuffer) X(GetSemaphoreCounterValue) X(QueueSubmit2) X(ResetDescriptorPool)      \
    X(UpdateDescriptorSets) X(WaitSemaphores)

#define VKD_DECLARE(n) extern PFN_vk##n vkd_##n;
VKD_FUNCS(VKD_DECLARE)
#undef VKD_DECLARE
void vkd_load(VkDevice dev);

#ifndef VKD_NO_REDIRECT
#include "vkq.h"  // (command recording: deferred to the recording thread while a list is translated)
#define vkAllocateDescriptorSets vkd_AllocateDescriptorSets
#define vkBeginCommandBuffer vkq_BeginCommandBuffer
#define vkCmdBeginQuery vkq_CmdBeginQuery
#define vkCmdBeginRendering vkq_CmdBeginRendering
#define vkCmdBindDescriptorSets vkq_CmdBindDescriptorSets
#define vkCmdBindIndexBuffer vkq_CmdBindIndexBuffer
#define vkCmdBindPipeline vkq_CmdBindPipeline
#define vkCmdBindVertexBuffers vkq_CmdBindVertexBuffers
#define vkCmdBlitImage vkq_CmdBlitImage
#define vkCmdClearAttachments vkq_CmdClearAttachments
#define vkCmdClearColorImage vkq_CmdClearColorImage
#define vkCmdClearDepthStencilImage vkq_CmdClearDepthStencilImage
#define vkCmdCopyBuffer vkq_CmdCopyBuffer
#define vkCmdCopyBufferToImage vkq_CmdCopyBufferToImage
#define vkCmdCopyImage vkq_CmdCopyImage
#define vkCmdCopyImageToBuffer vkq_CmdCopyImageToBuffer
#define vkCmdDispatch vkq_CmdDispatch
#define vkCmdDraw vkq_CmdDraw
#define vkCmdDrawIndexed vkq_CmdDrawIndexed
#define vkCmdEndQuery vkq_CmdEndQuery
#define vkCmdEndRendering vkq_CmdEndRendering
#define vkCmdPipelineBarrier2 vkq_CmdPipelineBarrier2
#define vkCmdPushConstants vkq_CmdPushConstants
#define vkCmdResetQueryPool vkq_CmdResetQueryPool
#define vkCmdResolveImage vkq_CmdResolveImage
#define vkCmdSetBlendConstants vkq_CmdSetBlendConstants
#define vkCmdSetCullMode vkq_CmdSetCullMode
#define vkCmdSetDepthBias vkq_CmdSetDepthBias
#define vkCmdSetDepthBiasEnable vkq_CmdSetDepthBiasEnable
#define vkCmdSetDepthCompareOp vkq_CmdSetDepthCompareOp
#define vkCmdSetDepthTestEnable vkq_CmdSetDepthTestEnable
#define vkCmdSetDepthWriteEnable vkq_CmdSetDepthWriteEnable
#define vkCmdSetFrontFace vkq_CmdSetFrontFace
#define vkCmdSetLineWidth vkq_CmdSetLineWidth
#define vkCmdSetPrimitiveRestartEnable vkq_CmdSetPrimitiveRestartEnable
#define vkCmdSetPrimitiveTopology vkq_CmdSetPrimitiveTopology
#define vkCmdSetScissor vkq_CmdSetScissor
#define vkCmdSetStencilCompareMask vkq_CmdSetStencilCompareMask
#define vkCmdSetStencilOp vkq_CmdSetStencilOp
#define vkCmdSetStencilReference vkq_CmdSetStencilReference
#define vkCmdSetStencilTestEnable vkq_CmdSetStencilTestEnable
#define vkCmdSetStencilWriteMask vkq_CmdSetStencilWriteMask
#define vkCmdSetViewport vkq_CmdSetViewport
#define vkCmdWriteTimestamp2 vkq_CmdWriteTimestamp2
#define vkEndCommandBuffer vkq_EndCommandBuffer
#define vkGetSemaphoreCounterValue vkd_GetSemaphoreCounterValue
#define vkQueueSubmit2 vkq_QueueSubmit2
#define vkResetDescriptorPool vkd_ResetDescriptorPool
#define vkUpdateDescriptorSets vkd_UpdateDescriptorSets
#define vkWaitSemaphores vkq_WaitSemaphores
#endif
