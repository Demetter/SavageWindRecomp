// Real GPU roundtrip: every 16-bit value, three array layers, nine mip levels,
// and a second update. No shader approximation, resolution change, or CPU texel conversion.
#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../runtime/nvn/vulkan/depth_mirror.h"
#define CHECK(call) do { VkResult r_ = (call); if (r_ != VK_SUCCESS) { \
    fprintf(stderr, "%s failed: %d\n", #call, r_); exit(1); } } while (0)

static uint32_t memory_type(VkPhysicalDevice physical, uint32_t bits, VkMemoryPropertyFlags flags) {
    VkPhysicalDeviceMemoryProperties p;
    vkGetPhysicalDeviceMemoryProperties(physical, &p);
    for (uint32_t i = 0; i < p.memoryTypeCount; i++)
        if ((bits & (1u << i)) && (p.memoryTypes[i].propertyFlags & flags) == flags) return i;
    assert(!"required memory type unavailable"); return 0;
}
static VkBuffer make_buffer(VkPhysicalDevice physical, VkDevice device, VkDeviceSize size,
                            VkDeviceMemory *memory) {
    VkBufferCreateInfo bi = {.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size=size,
        .usage=VK_BUFFER_USAGE_TRANSFER_SRC_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    VkBuffer buffer; CHECK(vkCreateBuffer(device, &bi, NULL, &buffer));
    VkMemoryRequirements req; vkGetBufferMemoryRequirements(device, buffer, &req);
    VkMemoryAllocateInfo ai = {.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize=req.size,
        .memoryTypeIndex=memory_type(physical, req.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};
    CHECK(vkAllocateMemory(device, &ai, NULL, memory));
    CHECK(vkBindBufferMemory(device, buffer, *memory, 0)); return buffer;
}
static VkImage make_image(VkPhysicalDevice physical, VkDevice device, VkFormat format,
                          VkDeviceMemory *memory) {
    VkImageCreateInfo ii = {.sType=VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType=VK_IMAGE_TYPE_2D,
        .format=format, .extent={256,256,1}, .mipLevels=9, .arrayLayers=3,
        .samples=VK_SAMPLE_COUNT_1_BIT, .tiling=VK_IMAGE_TILING_OPTIMAL,
        .usage=VK_IMAGE_USAGE_TRANSFER_SRC_BIT|VK_IMAGE_USAGE_TRANSFER_DST_BIT|VK_IMAGE_USAGE_SAMPLED_BIT};
    VkImage image; CHECK(vkCreateImage(device, &ii, NULL, &image));
    VkMemoryRequirements req; vkGetImageMemoryRequirements(device, image, &req);
    VkMemoryAllocateInfo ai = {.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize=req.size,
        .memoryTypeIndex=memory_type(physical, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
    CHECK(vkAllocateMemory(device, &ai, NULL, memory));
    CHECK(vkBindImageMemory(device, image, *memory, 0)); return image;
}
static void barrier(VkCommandBuffer cb) {
    VkMemoryBarrier2 b = {.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .srcStageMask=VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, .srcAccessMask=VK_ACCESS_2_MEMORY_WRITE_BIT,
        .dstStageMask=VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        .dstAccessMask=VK_ACCESS_2_MEMORY_READ_BIT|VK_ACCESS_2_MEMORY_WRITE_BIT};
    VkDependencyInfo d = {.sType=VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount=1, .pMemoryBarriers=&b};
    vkCmdPipelineBarrier2(cb, &d);
}
int main(int argc, char **argv) {
    bool fp32 = argc > 1 && !strcmp(argv[1], "--float");
    uint32_t bpp = fp32 ? 4 : 2;
    VkFormat source_format = fp32 ? VK_FORMAT_R32_SFLOAT : VK_FORMAT_R16_UNORM;
    VkFormat depth_format = fp32 ? VK_FORMAT_D32_SFLOAT : VK_FORMAT_D16_UNORM;
    VkBufferImageCopy color[9], depth[9];
    VkDeviceSize size = depth_mirror_regions(256,256,3,9,bpp,VK_IMAGE_ASPECT_COLOR_BIT,color,9);
    assert(size && size == depth_mirror_regions(256,256,3,9,bpp,VK_IMAGE_ASPECT_DEPTH_BIT,depth,9));
    assert(!depth_mirror_regions(0,256,3,9,bpp,VK_IMAGE_ASPECT_COLOR_BIT,color,9));
    assert(!depth_mirror_regions(256,256,3,9,bpp,VK_IMAGE_ASPECT_COLOR_BIT,color,8));
    for (int i = 0; i < 9; i++) assert(!(color[i].bufferOffset & 3));
    VkApplicationInfo app = {.sType=VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion=VK_API_VERSION_1_3};
    VkInstanceCreateInfo ici = {.sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo=&app};
    VkInstance instance; CHECK(vkCreateInstance(&ici,NULL,&instance));
    VkPhysicalDevice devices[16], physical = VK_NULL_HANDLE; uint32_t n=16, family=0;
    CHECK(vkEnumeratePhysicalDevices(instance,&n,devices));
    for (uint32_t i=0; i<n && !physical; i++) {
        VkQueueFamilyProperties q[32]; uint32_t nq=32;
        vkGetPhysicalDeviceQueueFamilyProperties(devices[i],&nq,q);
        for (uint32_t j=0; j<nq; j++) if (q[j].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            physical=devices[i]; family=j; break;
        }
    }
    assert(physical);
    VkFormatProperties3 fp3 = {.sType=VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_3};
    VkFormatProperties2 fp2 = {.sType=VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2,.pNext=&fp3};
    vkGetPhysicalDeviceFormatProperties2(physical,depth_format,&fp2);
    assert(fp3.optimalTilingFeatures & VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_DEPTH_COMPARISON_BIT);
    VkPhysicalDeviceProperties properties; vkGetPhysicalDeviceProperties(physical,&properties);
    float priority=1;
    VkDeviceQueueCreateInfo qi = {.sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex=family,.queueCount=1,.pQueuePriorities=&priority};
    VkPhysicalDeviceVulkan13Features f13 = {.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,.synchronization2=VK_TRUE};
    VkDeviceCreateInfo di = {.sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,.pNext=&f13,.queueCreateInfoCount=1,.pQueueCreateInfos=&qi};
    const char *extension = VK_EXT_DEPTH_RANGE_UNRESTRICTED_EXTENSION_NAME;
    if (fp32) { di.enabledExtensionCount=1; di.ppEnabledExtensionNames=&extension; }
    VkDevice device; CHECK(vkCreateDevice(physical,&di,NULL,&device));
    VkQueue queue; vkGetDeviceQueue(device,family,0,&queue);
    VkDeviceMemory bm[3], im[2]; VkBuffer buffers[3]; uint8_t *mapped[3];
    for (int i=0; i<3; i++) {
        buffers[i]=make_buffer(physical,device,size,&bm[i]);
        CHECK(vkMapMemory(device,bm[i],0,size,0,(void **)&mapped[i]));
    }
    VkImage images[2] = {make_image(physical,device,source_format,&im[0]),
                         make_image(physical,device,depth_format,&im[1])};
    VkCommandPoolCreateInfo pi = {.sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .queueFamilyIndex=family,.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT};
    VkCommandPool pool; CHECK(vkCreateCommandPool(device,&pi,NULL,&pool));
    VkCommandBufferAllocateInfo ai = {.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool=pool,.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,.commandBufferCount=1};
    VkCommandBuffer cb; CHECK(vkAllocateCommandBuffers(device,&ai,&cb));
    for (unsigned round=0; round<2; round++) {
        memset(mapped[0],0x5a,(size_t)size); memset(mapped[2],0,(size_t)size);
        for (unsigned level=0; level<9; level++) {
            uint16_t *p=(uint16_t *)(mapped[0]+color[level].bufferOffset);
            size_t texels=(size_t)color[level].imageExtent.width*color[level].imageExtent.height*3;
            for (size_t i=0; i<texels; i++) {
                uint16_t u=(uint16_t)((i+level*919u) ^ (round?0xa5a5u:0));
                if (fp32) {
                    // Finite values on both sides of [0,1], with signed zero,
                    // subnormal, and extreme finite bit patterns at each mip.
                    static const uint32_t edge[]={0,0x80000000u,1,0x80000001u,0x3f800000u,0x7f7fffffu,0xff7fffffu};
                    float f=((int)u-32768)/129.0f;
                    if (i < sizeof edge/sizeof *edge) memcpy(&f,&edge[i],4);
                    memcpy((uint8_t *)p+i*4,&f,4);
                } else p[i]=u;
            }
        }
        CHECK(vkResetCommandBuffer(cb,0));
        VkCommandBufferBeginInfo begin={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        CHECK(vkBeginCommandBuffer(cb,&begin));
        if (!round) for (int i=0;i<2;i++) {
            VkImageMemoryBarrier2 ib={.sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                .dstStageMask=VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                .dstAccessMask=VK_ACCESS_2_MEMORY_READ_BIT|VK_ACCESS_2_MEMORY_WRITE_BIT,
                .oldLayout=VK_IMAGE_LAYOUT_UNDEFINED,.newLayout=VK_IMAGE_LAYOUT_GENERAL,
                .srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,
                .image=images[i],.subresourceRange={i?VK_IMAGE_ASPECT_DEPTH_BIT:VK_IMAGE_ASPECT_COLOR_BIT,0,9,0,3}};
            VkDependencyInfo dep={.sType=VK_STRUCTURE_TYPE_DEPENDENCY_INFO,.imageMemoryBarrierCount=1,.pImageMemoryBarriers=&ib};
            vkCmdPipelineBarrier2(cb,&dep);
        }
        barrier(cb);
        vkCmdCopyBufferToImage(cb,buffers[0],images[0],VK_IMAGE_LAYOUT_GENERAL,9,color);
        barrier(cb);
        vkCmdCopyImageToBuffer(cb,images[0],VK_IMAGE_LAYOUT_GENERAL,buffers[1],9,color);
        barrier(cb);
        vkCmdCopyBufferToImage(cb,buffers[1],images[1],VK_IMAGE_LAYOUT_GENERAL,9,depth);
        barrier(cb);
        vkCmdCopyImageToBuffer(cb,images[1],VK_IMAGE_LAYOUT_GENERAL,buffers[2],9,depth);
        VkMemoryBarrier2 host={.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
            .srcStageMask=VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,.srcAccessMask=VK_ACCESS_2_MEMORY_WRITE_BIT,
            .dstStageMask=VK_PIPELINE_STAGE_2_HOST_BIT,.dstAccessMask=VK_ACCESS_2_HOST_READ_BIT};
        VkDependencyInfo dep={.sType=VK_STRUCTURE_TYPE_DEPENDENCY_INFO,.memoryBarrierCount=1,.pMemoryBarriers=&host};
        vkCmdPipelineBarrier2(cb,&dep);
        CHECK(vkEndCommandBuffer(cb));
        VkSubmitInfo si={.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,.commandBufferCount=1,.pCommandBuffers=&cb};
        CHECK(vkQueueSubmit(queue,1,&si,VK_NULL_HANDLE)); CHECK(vkQueueWaitIdle(queue));
        for (int level=0;level<9;level++) {
            size_t bytes=(size_t)color[level].imageExtent.width*color[level].imageExtent.height*3*bpp;
            assert(!memcmp(mapped[0]+color[level].bufferOffset,mapped[2]+depth[level].bufferOffset,bytes));
        }
    }
    printf("%s: %s preserved test values, 3 layers, 9 mips, and a second update.\n",properties.deviceName,
        fp32?"R32 to D32 (including finite values outside [0,1])":"R16 to D16 (all 65536 values)");
    vkDestroyCommandPool(device,pool,NULL);
    for(int i=0;i<2;i++){vkDestroyImage(device,images[i],NULL);vkFreeMemory(device,im[i],NULL);}
    for(int i=0;i<3;i++){vkUnmapMemory(device,bm[i]);vkDestroyBuffer(device,buffers[i],NULL);vkFreeMemory(device,bm[i],NULL);}
    vkDestroyDevice(device,NULL);vkDestroyInstance(instance,NULL);return 0;
}
