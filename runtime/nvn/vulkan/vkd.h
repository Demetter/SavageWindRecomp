// The submission thread's Vulkan calls, straight to the driver. Called through vulkan-1.dll, each one went through
// the loader's trampoline (its dispatch table looked up from the handle) first: ~10% of the submission thread in
// BOTW's open world, at ~20 calls a draw and ~5,000 draws a frame. vkd_load (after vkCreateDevice) takes each
// function's device-level entry point from vkGetDeviceProcAddr, and vk.h's names below call those; a function the
// device doesn't give keeps the loader's. vkd.c, which defines them, sees the loader's names (VKD_NO_REDIRECT).
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
#define vkAllocateDescriptorSets vkd_AllocateDescriptorSets
#define vkBeginCommandBuffer vkd_BeginCommandBuffer
#define vkCmdBeginQuery vkd_CmdBeginQuery
#define vkCmdBeginRendering vkd_CmdBeginRendering
#define vkCmdBindDescriptorSets vkd_CmdBindDescriptorSets
#define vkCmdBindIndexBuffer vkd_CmdBindIndexBuffer
#define vkCmdBindPipeline vkd_CmdBindPipeline
#define vkCmdBindVertexBuffers vkd_CmdBindVertexBuffers
#define vkCmdBlitImage vkd_CmdBlitImage
#define vkCmdClearAttachments vkd_CmdClearAttachments
#define vkCmdClearColorImage vkd_CmdClearColorImage
#define vkCmdClearDepthStencilImage vkd_CmdClearDepthStencilImage
#define vkCmdCopyBuffer vkd_CmdCopyBuffer
#define vkCmdCopyBufferToImage vkd_CmdCopyBufferToImage
#define vkCmdCopyImage vkd_CmdCopyImage
#define vkCmdCopyImageToBuffer vkd_CmdCopyImageToBuffer
#define vkCmdDispatch vkd_CmdDispatch
#define vkCmdDraw vkd_CmdDraw
#define vkCmdDrawIndexed vkd_CmdDrawIndexed
#define vkCmdEndQuery vkd_CmdEndQuery
#define vkCmdEndRendering vkd_CmdEndRendering
#define vkCmdPipelineBarrier2 vkd_CmdPipelineBarrier2
#define vkCmdPushConstants vkd_CmdPushConstants
#define vkCmdResetQueryPool vkd_CmdResetQueryPool
#define vkCmdResolveImage vkd_CmdResolveImage
#define vkCmdSetBlendConstants vkd_CmdSetBlendConstants
#define vkCmdSetCullMode vkd_CmdSetCullMode
#define vkCmdSetDepthBias vkd_CmdSetDepthBias
#define vkCmdSetDepthBiasEnable vkd_CmdSetDepthBiasEnable
#define vkCmdSetDepthCompareOp vkd_CmdSetDepthCompareOp
#define vkCmdSetDepthTestEnable vkd_CmdSetDepthTestEnable
#define vkCmdSetDepthWriteEnable vkd_CmdSetDepthWriteEnable
#define vkCmdSetFrontFace vkd_CmdSetFrontFace
#define vkCmdSetLineWidth vkd_CmdSetLineWidth
#define vkCmdSetPrimitiveRestartEnable vkd_CmdSetPrimitiveRestartEnable
#define vkCmdSetPrimitiveTopology vkd_CmdSetPrimitiveTopology
#define vkCmdSetScissor vkd_CmdSetScissor
#define vkCmdSetStencilCompareMask vkd_CmdSetStencilCompareMask
#define vkCmdSetStencilOp vkd_CmdSetStencilOp
#define vkCmdSetStencilReference vkd_CmdSetStencilReference
#define vkCmdSetStencilTestEnable vkd_CmdSetStencilTestEnable
#define vkCmdSetStencilWriteMask vkd_CmdSetStencilWriteMask
#define vkCmdSetViewport vkd_CmdSetViewport
#define vkCmdWriteTimestamp2 vkd_CmdWriteTimestamp2
#define vkEndCommandBuffer vkd_EndCommandBuffer
#define vkGetSemaphoreCounterValue vkd_GetSemaphoreCounterValue
#define vkQueueSubmit2 vkd_QueueSubmit2
#define vkResetDescriptorPool vkd_ResetDescriptorPool
#define vkUpdateDescriptorSets vkd_UpdateDescriptorSets
#define vkWaitSemaphores vkd_WaitSemaphores
#endif
