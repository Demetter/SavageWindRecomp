// Execute the original and literal-specialized shader on the same GPU. Checks
// exact integer/NaN/negative-zero payloads plus finite floating-point arithmetic.
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>
#include <shaderc/shaderc.h>
#include "../runtime/nvn/vulkan/shader_constants.h"
#define CHECK(call) do { VkResult result_ = (call); if (result_ != VK_SUCCESS) { fprintf(stderr, "%s: %d\n", #call, result_); exit(1); } } while (0)

int main(void) {
    const uint32_t input[16] = {0x80000000, 0x7fc12345, 0xffffffff, 0,
        0x3fa00000, 0xc0000000, 0x3f000000, 0x7f800000,
        1, 0x7fffff, 0x800000, 0xff800000, 0x12345678, 42, 0x80000001, 0x7fffffff};
    char source[8192];
    size_t len = (size_t)snprintf(source, sizeof source,
        "#version 460\nlayout(local_size_x=1) in;\n"
        "layout(std140, set = 0, binding = 1) uniform CB1 { uvec4 cb1[4]; };\n"
        "layout(std430, set=0, binding=17) buffer Out {uint v[];} result_;\nvoid main(){\n");
    for (unsigned i = 0; i < 16; i++)
        len += (size_t)snprintf(source + len, sizeof source - len, "result_.v[%u]=cb1[%u][%u];\n", i, i / 4, i % 4);
    snprintf(source + len, sizeof source - len,
        "result_.v[16]=floatBitsToUint(fma(uintBitsToFloat(cb1[1][0]),uintBitsToFloat(cb1[1][1]),uintBitsToFloat(cb1[1][2])));}\n");
    char *folded = vk_shader_literal_constants(source, (const uint8_t *)input, sizeof input);
    assert(folded && !strstr(folded, "uniform CB1"));
    VkInstance instance;
    VkApplicationInfo app = {.sType=VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion=VK_API_VERSION_1_3};
    VkInstanceCreateInfo ici = {.sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo=&app};
    CHECK(vkCreateInstance(&ici, NULL, &instance));
    VkPhysicalDevice devices[16], physical = VK_NULL_HANDLE;
    uint32_t ndev = 16, family = 0;
    CHECK(vkEnumeratePhysicalDevices(instance, &ndev, devices));
    for (unsigned d = 0; d < ndev && !physical; d++) {
        VkQueueFamilyProperties props[32]; uint32_t n = 32;
        vkGetPhysicalDeviceQueueFamilyProperties(devices[d], &n, props);
        for (unsigned q = 0; q < n; q++) if (props[q].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            physical = devices[d]; family = q; break;
        }
    }
    assert(physical);
    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(physical, &properties);
    float priority = 1;
    VkDeviceQueueCreateInfo qci = {.sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex=family,
        .queueCount=1, .pQueuePriorities=&priority};
    VkPhysicalDeviceVulkan13Features f13 = {.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, .maintenance4=VK_TRUE};
    VkDeviceCreateInfo dci = {.sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext=&f13, .queueCreateInfoCount=1, .pQueueCreateInfos=&qci};
    VkDevice device; CHECK(vkCreateDevice(physical, &dci, NULL, &device));
    VkQueue queue; vkGetDeviceQueue(device, family, 0, &queue);
    VkDeviceSize output_offset = properties.limits.minStorageBufferOffsetAlignment;
    if (output_offset < 256) output_offset = 256;
    VkBufferCreateInfo bci = {.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size=output_offset+128,
        .usage=VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT|VK_BUFFER_USAGE_STORAGE_BUFFER_BIT};
    VkBuffer buffer; CHECK(vkCreateBuffer(device, &bci, NULL, &buffer));
    VkMemoryRequirements req; vkGetBufferMemoryRequirements(device, buffer, &req);
    VkPhysicalDeviceMemoryProperties memprops; vkGetPhysicalDeviceMemoryProperties(physical, &memprops);
    uint32_t type = 0;
    for (; type < memprops.memoryTypeCount; type++)
        if ((req.memoryTypeBits & (1u << type)) &&
            (memprops.memoryTypes[type].propertyFlags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
            (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) break;
    assert(type < memprops.memoryTypeCount);
    VkMemoryAllocateInfo mai = {.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize=req.size, .memoryTypeIndex=type};
    VkDeviceMemory memory; CHECK(vkAllocateMemory(device, &mai, NULL, &memory));
    CHECK(vkBindBufferMemory(device, buffer, memory, 0));
    uint8_t *mapped; CHECK(vkMapMemory(device, memory, 0, req.size, 0, (void **)&mapped));
    memcpy(mapped, input, sizeof input);
    VkCommandPoolCreateInfo pci = {.sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex=family,
        .flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT};
    VkCommandPool pool; CHECK(vkCreateCommandPool(device, &pci, NULL, &pool));
    VkCommandBufferAllocateInfo cai = {.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool=pool,
        .level=VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount=1};
    VkCommandBuffer cmd; CHECK(vkAllocateCommandBuffers(device, &cai, &cmd));
    shaderc_compiler_t compiler = shaderc_compiler_initialize();
    shaderc_compile_options_t options = shaderc_compile_options_initialize();
    shaderc_compile_options_set_target_env(options, shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_3);
    shaderc_compile_options_set_optimization_level(options, shaderc_optimization_level_performance);
    uint32_t results[2][17];
    for (unsigned variant = 0; variant < 2; variant++) {
        const char *text = variant ? folded : source;
        shaderc_compilation_result_t compiled = shaderc_compile_into_spv(compiler, text, strlen(text),
            shaderc_compute_shader, "literal-constants", "main", options);
        if (shaderc_result_get_compilation_status(compiled) != shaderc_compilation_status_success) {
            fprintf(stderr, "%s\n", shaderc_result_get_error_message(compiled)); return 1;
        }
        VkShaderModuleCreateInfo smci = {.sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize=shaderc_result_get_length(compiled), .pCode=(const uint32_t *)shaderc_result_get_bytes(compiled)};
        VkShaderModule module; CHECK(vkCreateShaderModule(device, &smci, NULL, &module));
        VkDescriptorSetLayoutBinding bindings[2] = {
            {17,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,NULL},
            {1,VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,NULL}};
        VkDescriptorSetLayoutCreateInfo slci = {.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount=variant?1:2, .pBindings=bindings};
        VkDescriptorSetLayout sl; CHECK(vkCreateDescriptorSetLayout(device, &slci, NULL, &sl));
        VkPipelineLayoutCreateInfo plci = {.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount=1, .pSetLayouts=&sl};
        VkPipelineLayout layout; CHECK(vkCreatePipelineLayout(device, &plci, NULL, &layout));
        VkComputePipelineCreateInfo cpci = {.sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, .layout=layout,
            .stage={.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage=VK_SHADER_STAGE_COMPUTE_BIT,
                .module=module, .pName="main"}};
        VkPipeline pipeline; CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, NULL, &pipeline));
        VkDescriptorPoolSize sizes[2] = {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1},{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1}};
        VkDescriptorPoolCreateInfo dpci = {.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .maxSets=1, .poolSizeCount=2, .pPoolSizes=sizes};
        VkDescriptorPool dp; CHECK(vkCreateDescriptorPool(device, &dpci, NULL, &dp));
        VkDescriptorSetAllocateInfo dsai = {.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool=dp, .descriptorSetCount=1, .pSetLayouts=&sl};
        VkDescriptorSet ds; CHECK(vkAllocateDescriptorSets(device, &dsai, &ds));
        VkDescriptorBufferInfo bi[2] = {{buffer,output_offset,128},{buffer,0,sizeof input}};
        VkWriteDescriptorSet writes[2];
        for (unsigned i = 0; i < 2; i++) writes[i] = (VkWriteDescriptorSet){.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet=ds, .dstBinding=bindings[i].binding, .descriptorCount=1, .descriptorType=bindings[i].descriptorType, .pBufferInfo=&bi[i]};
        vkUpdateDescriptorSets(device, variant?1:2, writes, 0, NULL);
        CHECK(vkResetCommandBuffer(cmd, 0));
        VkCommandBufferBeginInfo begin = {.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        CHECK(vkBeginCommandBuffer(cmd, &begin));
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &ds, 0, NULL);
        vkCmdDispatch(cmd, 1, 1, 1);
        VkMemoryBarrier barrier = {.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask=VK_ACCESS_HOST_READ_BIT};
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0, NULL, 0, NULL);
        CHECK(vkEndCommandBuffer(cmd));
        VkSubmitInfo submit = {.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount=1, .pCommandBuffers=&cmd};
        CHECK(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE));
        CHECK(vkQueueWaitIdle(queue));
        memcpy(results[variant], mapped + output_offset, sizeof results[0]);
        vkDestroyDescriptorPool(device, dp, NULL); vkDestroyPipeline(device, pipeline, NULL);
        vkDestroyPipelineLayout(device, layout, NULL); vkDestroyDescriptorSetLayout(device, sl, NULL);
        vkDestroyShaderModule(device, module, NULL); shaderc_result_release(compiled);
    }
    assert(!memcmp(results[0], input, sizeof input));
    assert(results[0][16] == 0xc0000000u);
    assert(!memcmp(results[0], results[1], sizeof results[0]));
    printf("%s: original and literal shader produced identical 17 words; specialized shader has no c1 binding.\n", properties.deviceName);
    shaderc_compile_options_release(options); shaderc_compiler_release(compiler);
    vkDestroyCommandPool(device, pool, NULL); vkUnmapMemory(device, memory);
    vkDestroyBuffer(device, buffer, NULL); vkFreeMemory(device, memory, NULL);
    vkDestroyDevice(device, NULL); vkDestroyInstance(instance, NULL); free(folded);
    return 0;
}
