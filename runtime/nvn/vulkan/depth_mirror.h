#pragma once
#include <stdint.h>
#include <vulkan/vulkan.h>

// R16_UNORM/D16_UNORM and R32_SFLOAT/D32_SFLOAT have identical texel representations.
// A buffer bridge copies the bits without shader arithmetic or resampling and
// works without maintenance8's optional color/depth image-copy support.
// Offsets are four-byte aligned, including the small mip levels.
static VkDeviceSize depth_mirror_regions(uint32_t width, uint32_t height, uint32_t layers,
                                        uint32_t levels, uint32_t bytes_per_texel, VkImageAspectFlags aspect,
                                        VkBufferImageCopy *regions, uint32_t capacity) {
    if (!width || !height || !layers || !levels || levels > capacity || levels > 32 ||
        (bytes_per_texel != 2 && bytes_per_texel != 4)) return 0;
    VkDeviceSize size = 0;
    for (uint32_t level = 0; level < levels; level++) {
        uint32_t w = width >> level, h = height >> level;
        if (!w) w = 1;
        if (!h) h = 1;
        size = (size + 3) & ~(VkDeviceSize)3;
        regions[level] = (VkBufferImageCopy){.bufferOffset = size,
            .imageSubresource = {aspect, level, 0, layers}, .imageExtent = {w, h, 1}};
        size += (VkDeviceSize)w * h * layers * bytes_per_texel;
    }
    return size;
}
