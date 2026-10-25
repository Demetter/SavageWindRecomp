// Exact offscreen comparison: 64 UBO-bound draws vs one native multi-draw.
// Exercises VS->FS draw IDs, a nonzero table base, separate per-stage constants,
// native buffer addresses and all 4096 RGBA32_UINT pixels against a CPU oracle.
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>
#include <shaderc/shaderc.h>
#include "../runtime/nvn/vulkan/native_draw_constants.h"
#include "../runtime/nvn/vulkan/native_draw_geometry.h"
#define CHECK(x) do {VkResult r_=(x);if(r_!=VK_SUCCESS){fprintf(stderr,"%s: %d\n",#x,r_);exit(1);}}while(0)
enum { WIDTH=64, HEIGHT=64, DRAWS=64, FIRST=3 };
static uint32_t memory_type(VkPhysicalDevice p,uint32_t bits,VkMemoryPropertyFlags flags){
    VkPhysicalDeviceMemoryProperties m;vkGetPhysicalDeviceMemoryProperties(p,&m);
    for(uint32_t i=0;i<m.memoryTypeCount;i++)if((bits&(1u<<i))&&(m.memoryTypes[i].propertyFlags&flags)==flags)return i;
    fprintf(stderr,"Required memory type unavailable\n");exit(1);
}
static VkShaderModule compile(VkDevice d,shaderc_compiler_t c,shaderc_compile_options_t o,const char *text,bool fragment){
    shaderc_compilation_result_t r=shaderc_compile_into_spv(c,text,strlen(text),fragment?shaderc_fragment_shader:shaderc_vertex_shader,"native-multi-draw","main",o);
    if(shaderc_result_get_compilation_status(r)!=shaderc_compilation_status_success){fprintf(stderr,"%s\n",shaderc_result_get_error_message(r));exit(1);}
    VkShaderModuleCreateInfo ci={.sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,.codeSize=shaderc_result_get_length(r),.pCode=(const uint32_t*)shaderc_result_get_bytes(r)};
    VkShaderModule m;CHECK(vkCreateShaderModule(d,&ci,NULL,&m));shaderc_result_release(r);return m;
}
int main(void){
    const char *vs="#version 460\n"
        "#define cbr(cb,off) cb[min((off)>>4,uint(cb.length()-1))][((off)>>2)&3u]\n"
        "layout(std140, set = 0, binding = 3) uniform CB3 { uvec4 cb3[8]; };\n"
        "invariant gl_Position; precise gl_Position;\nvoid main() {\n"
        "uint off=uint(gl_VertexIndex)*16u; gl_Position=uintBitsToFloat(uvec4(cbr(cb3,off),cbr(cb3,off+4u),cbr(cb3,off+8u),cbr(cb3,off+12u)));}\n";
    const char *fs="#version 460\nlayout(std140, set = 1, binding = 4) uniform CB4 { uvec4 cb4[1]; };\n"
        "layout(location = 0) out uvec4 color;\nvoid main() {\ncolor=cb4[0];}\n";
    uint32_t masks[2];char *native_vs=vk_shader_native_draw_graphics(vs,0,1,&masks[0]);char *native_fs=vk_shader_native_draw_graphics(fs,1,1,&masks[1]);
    assert(native_vs&&native_fs&&masks[0]==8&&masks[1]==16);
    const char *geometry_vs="#version 460\nlayout(location = 0) in vec4 position;\n"
        "layout(std140, set = 0, binding = 3) uniform CB3 { uvec4 cb3[8]; };\n"
        "invariant gl_Position; precise gl_Position;\nvoid main() {\n"
        "gl_Position=position+uintBitsToFloat(cb3[6]);}\n";
    char *native_geometry_vs=vk_shader_native_draw_graphics(geometry_vs,0,1,&masks[0]);assert(native_geometry_vs&&masks[0]==8);
    VkApplicationInfo app={.sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,.apiVersion=VK_API_VERSION_1_3};
    VkInstanceCreateInfo ici={.sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,.pApplicationInfo=&app};VkInstance instance;CHECK(vkCreateInstance(&ici,NULL,&instance));
    VkPhysicalDevice devices[16],physical=VK_NULL_HANDLE;uint32_t ndev=16,family=0;CHECK(vkEnumeratePhysicalDevices(instance,&ndev,devices));
    for(unsigned d=0;d<ndev&&!physical;d++){VkQueueFamilyProperties q[32];uint32_t n=32;vkGetPhysicalDeviceQueueFamilyProperties(devices[d],&n,q);for(unsigned i=0;i<n;i++)if(q[i].queueFlags&VK_QUEUE_GRAPHICS_BIT){physical=devices[d];family=i;break;}}
    assert(physical);VkPhysicalDeviceProperties properties;vkGetPhysicalDeviceProperties(physical,&properties);
    VkPhysicalDeviceVulkan13Features have13={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceVulkan12Features have12={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,.pNext=&have13};
    VkPhysicalDeviceVulkan11Features have11={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,.pNext=&have12};
    VkPhysicalDeviceFeatures2 have={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,.pNext=&have11};vkGetPhysicalDeviceFeatures2(physical,&have);
    assert(have11.shaderDrawParameters&&have12.bufferDeviceAddress&&have13.dynamicRendering&&have.features.multiDrawIndirect&&have.features.shaderInt64&&properties.limits.maxDrawIndirectCount>=DRAWS);
    float priority=1;VkDeviceQueueCreateInfo qci={.sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,.queueFamilyIndex=family,.queueCount=1,.pQueuePriorities=&priority};
    VkPhysicalDeviceVulkan13Features f13={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,.dynamicRendering=VK_TRUE};
    VkPhysicalDeviceVulkan12Features f12={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,.pNext=&f13,.bufferDeviceAddress=VK_TRUE};
    VkPhysicalDeviceVulkan11Features f11={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,.pNext=&f12,.shaderDrawParameters=VK_TRUE};
    VkPhysicalDeviceFeatures features={.shaderInt64=VK_TRUE,.multiDrawIndirect=VK_TRUE,.robustBufferAccess=have.features.robustBufferAccess};
    VkDeviceCreateInfo dci={.sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,.pNext=&f11,.pEnabledFeatures=&features,.queueCreateInfoCount=1,.pQueueCreateInfos=&qci};VkDevice device;CHECK(vkCreateDevice(physical,&dci,NULL,&device));VkQueue queue;vkGetDeviceQueue(device,family,0,&queue);
    VkDeviceSize stride=properties.limits.minUniformBufferOffsetAlignment;if(stride<256)stride=256;
    VkDeviceSize table_offset=DRAWS*stride*2,alignment=properties.limits.minStorageBufferOffsetAlignment;
    table_offset=(table_offset+alignment-1)/alignment*alignment;
    VkDeviceSize indirect_offset=table_offset+(DRAWS+FIRST)*sizeof(NativeDrawConstants);
    VkDeviceSize indexed_offset=indirect_offset+DRAWS*sizeof(VkDrawIndirectCommand),indices_offset=indexed_offset+DRAWS*sizeof(VkDrawIndexedIndirectCommand);
    VkDeviceSize output_offset=indices_offset+DRAWS*16,output_size=WIDTH*HEIGHT*16;
    VkBufferCreateInfo bci={.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,.size=output_offset+output_size,.usage=VK_BUFFER_USAGE_VERTEX_BUFFER_BIT|VK_BUFFER_USAGE_INDEX_BUFFER_BIT|VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT|VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT|VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    VkBuffer buffer;CHECK(vkCreateBuffer(device,&bci,NULL,&buffer));VkMemoryRequirements req;vkGetBufferMemoryRequirements(device,buffer,&req);
    VkMemoryAllocateFlagsInfo flags={.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,.flags=VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT};
    VkMemoryAllocateInfo mai={.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,.pNext=&flags,.allocationSize=req.size,.memoryTypeIndex=memory_type(physical,req.memoryTypeBits,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};
    VkDeviceMemory memory;CHECK(vkAllocateMemory(device,&mai,NULL,&memory));CHECK(vkBindBufferMemory(device,buffer,memory,0));uint8_t *mapped;CHECK(vkMapMemory(device,memory,0,req.size,0,(void**)&mapped));memset(mapped,0,(size_t)(output_offset+output_size));
    VkBufferDeviceAddressInfo ai={.sType=VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,.buffer=buffer};VkDeviceAddress address=vkGetBufferDeviceAddress(device,&ai);assert(address);
    NativeDrawConstants *table=(void*)(mapped+table_offset);VkDrawIndirectCommand *draws=(void*)(mapped+indirect_offset);
    uint32_t colors[DRAWS][4];
    for(unsigned d=0;d<DRAWS;d++){
        float x=-1.f+(float)(d%8)*.25f,y=-1.f+(float)(d/8)*.25f;
        float positions[6][4]={{x,y,0.5f,1},{x+.25f,y,0.5f,1},{x,y+.25f,0.5f,1},{x+.25f,y,0.5f,1},{x+.25f,y+.25f,0.5f,1},{x,y+.25f,0.5f,1}};
        colors[d][0]=d+1;colors[d][1]=0x80000000u^d;colors[d][2]=0x7fc12345u+d;colors[d][3]=0xffffffffu-d;
        memcpy(mapped+d*stride*2,positions,sizeof positions);memcpy(mapped+d*stride*2+stride,colors[d],16);
        table[d+FIRST].constants[3]=(NativeDrawConstant){address+d*stride*2,24,0};
        table[d+FIRST].constants[18+4]=(NativeDrawConstant){address+d*stride*2+stride,4,0};
        draws[d]=(VkDrawIndirectCommand){6,1,0,0};
        uint16_t *indices=(void*)(mapped+indices_offset+d*16);for(unsigned i=0;i<6;i++)indices[i]=(uint16_t)i;
    }
    VkFormat format=VK_FORMAT_R32G32B32A32_UINT;
    VkImageCreateInfo imci={.sType=VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,.imageType=VK_IMAGE_TYPE_2D,.format=format,.extent={WIDTH,HEIGHT,1},.mipLevels=1,.arrayLayers=1,.samples=VK_SAMPLE_COUNT_1_BIT,.tiling=VK_IMAGE_TILING_OPTIMAL,.usage=VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_TRANSFER_SRC_BIT,.sharingMode=VK_SHARING_MODE_EXCLUSIVE};
    VkImage image;CHECK(vkCreateImage(device,&imci,NULL,&image));vkGetImageMemoryRequirements(device,image,&req);
    VkMemoryAllocateInfo image_mai={.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,.allocationSize=req.size,.memoryTypeIndex=memory_type(physical,req.memoryTypeBits,0)};VkDeviceMemory image_memory;CHECK(vkAllocateMemory(device,&image_mai,NULL,&image_memory));CHECK(vkBindImageMemory(device,image,image_memory,0));
    VkImageViewCreateInfo ivci={.sType=VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,.image=image,.viewType=VK_IMAGE_VIEW_TYPE_2D,.format=format,.subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1}};VkImageView view;CHECK(vkCreateImageView(device,&ivci,NULL,&view));
    VkCommandPoolCreateInfo cpci={.sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,.queueFamilyIndex=family,.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT};VkCommandPool pool;CHECK(vkCreateCommandPool(device,&cpci,NULL,&pool));
    VkCommandBufferAllocateInfo cai={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,.commandPool=pool,.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,.commandBufferCount=1};VkCommandBuffer cmd;CHECK(vkAllocateCommandBuffers(device,&cai,&cmd));
    shaderc_compiler_t compiler=shaderc_compiler_initialize();shaderc_compile_options_t options=shaderc_compile_options_initialize();shaderc_compile_options_set_target_env(options,shaderc_target_env_vulkan,shaderc_env_version_vulkan_1_3);shaderc_compile_options_set_optimization_level(options,shaderc_optimization_level_performance);
    for (unsigned variant=0; variant<6; variant++) {
        unsigned native=variant&1,indexed=variant>=2,hybrid=variant>=4;
        // The draw order starts in the middle. Select the lowest shared binding
        // before emitting commands; negative rebases from the first draw's
        // binding would violate vertex-input robustness even at equal addresses.
        VkDeviceSize vertex_base=0;
        VkDrawIndexedIndirectCommand *indexed_draws=(void*)(mapped+indexed_offset);
        for(unsigned d=0;d<DRAWS;d++) {
            unsigned cell=indexed?(d*17+32)%DRAWS:d;
            table[d+FIRST].constants[3]=(NativeDrawConstant){address+cell*stride*2,32,0};
            table[d+FIRST].constants[18+4]=(NativeDrawConstant){address+cell*stride*2+stride,4,0};
            uint64_t old[17]={vertex_base},cur[17]={cell*stride*2};uint32_t strides[17]={16},divisors[17]={0};int32_t rebased=0;int64_t delta=0;
            if(!native_draw_vertex_rebase(1,old,cur,strides,divisors,0,&rebased,&delta))return 1;
            indexed_draws[d]=(VkDrawIndexedIndirectCommand){6,1,cell*8,rebased,0};
        }
        const char *vertex_source=indexed?(native?native_geometry_vs:geometry_vs):(native?native_vs:vs);
        VkShaderModule shaders[2]={compile(device,compiler,options,vertex_source,false),compile(device,compiler,options,native&&!hybrid?native_fs:fs,true)};
        VkDescriptorSetLayout layouts[3];
        for (unsigned set=0;set<3;set++) {
            VkDescriptorSetLayoutBinding binding={.binding=set==0?3:set==1?4:0,
                .descriptorType=set==2?VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                .descriptorCount=1,.stageFlags=set==0?VK_SHADER_STAGE_VERTEX_BIT:set==1?VK_SHADER_STAGE_FRAGMENT_BIT:VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_FRAGMENT_BIT};
            VkDescriptorSetLayoutCreateInfo li={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                .bindingCount=native&&(set==0||(set==1&&!hybrid))?0:1,.pBindings=&binding};
            CHECK(vkCreateDescriptorSetLayout(device,&li,NULL,&layouts[set]));
        }
        VkPushConstantRange push={.stageFlags=VK_SHADER_STAGE_VERTEX_BIT,.offset=0,.size=4};
        VkPipelineLayoutCreateInfo pli={.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,.setLayoutCount=3,.pSetLayouts=layouts,.pushConstantRangeCount=native?1:0,.pPushConstantRanges=&push};
        VkPipelineLayout layout;CHECK(vkCreatePipelineLayout(device,&pli,NULL,&layout));
        VkDescriptorPoolSize sizes[2]={{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,2},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1}};
        VkDescriptorPoolCreateInfo dpi={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,.maxSets=3,.poolSizeCount=2,.pPoolSizes=sizes};
        VkDescriptorPool descriptors;CHECK(vkCreateDescriptorPool(device,&dpi,NULL,&descriptors));
        VkDescriptorSetAllocateInfo dsa={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,.descriptorPool=descriptors,.descriptorSetCount=3,.pSetLayouts=layouts};
        VkDescriptorSet sets[3];CHECK(vkAllocateDescriptorSets(device,&dsa,sets));
        VkDescriptorBufferInfo bi[3]={{buffer,0,128},{buffer,stride,16},{buffer,table_offset,(DRAWS+FIRST)*sizeof(NativeDrawConstants)}};
        for (unsigned set=native?(hybrid?1:2):0;set<3;set++) {
            VkWriteDescriptorSet write={.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,.dstSet=sets[set],.dstBinding=set==0?3:set==1?4:0,
                .descriptorCount=1,.descriptorType=set==2?VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,.pBufferInfo=&bi[set]};
            vkUpdateDescriptorSets(device,1,&write,0,NULL);
        }
        VkPipelineShaderStageCreateInfo stages[2]={
            {.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,.stage=VK_SHADER_STAGE_VERTEX_BIT,.module=shaders[0],.pName="main"},
            {.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,.stage=VK_SHADER_STAGE_FRAGMENT_BIT,.module=shaders[1],.pName="main"}};
        VkVertexInputBindingDescription vertex_binding={0,16,VK_VERTEX_INPUT_RATE_VERTEX};
        VkVertexInputAttributeDescription vertex_attribute={0,0,VK_FORMAT_R32G32B32A32_SFLOAT,0};
        VkPipelineVertexInputStateCreateInfo vertex={.sType=VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            .vertexBindingDescriptionCount=indexed,.pVertexBindingDescriptions=&vertex_binding,.vertexAttributeDescriptionCount=indexed,.pVertexAttributeDescriptions=&vertex_attribute};
        VkPipelineInputAssemblyStateCreateInfo assembly={.sType=VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,.topology=VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
        VkViewport viewport={0,0,WIDTH,HEIGHT,0,1};VkRect2D scissor={{0,0},{WIDTH,HEIGHT}};
        VkPipelineViewportStateCreateInfo vp={.sType=VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,.viewportCount=1,.pViewports=&viewport,.scissorCount=1,.pScissors=&scissor};
        VkPipelineRasterizationStateCreateInfo raster={.sType=VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,.polygonMode=VK_POLYGON_MODE_FILL,.cullMode=VK_CULL_MODE_NONE,.frontFace=VK_FRONT_FACE_COUNTER_CLOCKWISE,.lineWidth=1};
        VkPipelineMultisampleStateCreateInfo multisample={.sType=VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,.rasterizationSamples=VK_SAMPLE_COUNT_1_BIT};
        VkPipelineColorBlendAttachmentState blend_attachment={.colorWriteMask=VK_COLOR_COMPONENT_R_BIT|VK_COLOR_COMPONENT_G_BIT|VK_COLOR_COMPONENT_B_BIT|VK_COLOR_COMPONENT_A_BIT};
        VkPipelineColorBlendStateCreateInfo blend={.sType=VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,.attachmentCount=1,.pAttachments=&blend_attachment};
        VkPipelineRenderingCreateInfo rendering={.sType=VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,.colorAttachmentCount=1,.pColorAttachmentFormats=&format};
        VkGraphicsPipelineCreateInfo pci={.sType=VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,.pNext=&rendering,.stageCount=2,.pStages=stages,
            .pVertexInputState=&vertex,.pInputAssemblyState=&assembly,.pViewportState=&vp,.pRasterizationState=&raster,.pMultisampleState=&multisample,.pColorBlendState=&blend,.layout=layout};
        VkPipeline pipeline;CHECK(vkCreateGraphicsPipelines(device,VK_NULL_HANDLE,1,&pci,NULL,&pipeline));
        CHECK(vkResetCommandBuffer(cmd,0));VkCommandBufferBeginInfo begin={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};CHECK(vkBeginCommandBuffer(cmd,&begin));
        VkImageMemoryBarrier image_barrier={.sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,.dstAccessMask=VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            .oldLayout=VK_IMAGE_LAYOUT_UNDEFINED,.newLayout=VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,.srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,
            .image=image,.subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1}};
        vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,0,0,NULL,0,NULL,1,&image_barrier);
        VkRenderingAttachmentInfo attachment={.sType=VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,.imageView=view,.imageLayout=VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,.loadOp=VK_ATTACHMENT_LOAD_OP_CLEAR,.storeOp=VK_ATTACHMENT_STORE_OP_STORE};
        VkRenderingInfo ri={.sType=VK_STRUCTURE_TYPE_RENDERING_INFO,.renderArea={{0,0},{WIDTH,HEIGHT}},.layerCount=1,.colorAttachmentCount=1,.pColorAttachments=&attachment};
        vkCmdBeginRendering(cmd,&ri);vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,pipeline);
        if (native) {
            uint32_t first=FIRST;
            uint32_t fragment_offset=0;
            vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,layout,0,3,sets,hybrid?1:0,&fragment_offset);
            vkCmdPushConstants(cmd,layout,VK_SHADER_STAGE_VERTEX_BIT,0,4,&first);
            if(indexed) {
                vkCmdBindVertexBuffers(cmd,0,1,&buffer,&vertex_base);vkCmdBindIndexBuffer(cmd,buffer,indices_offset,VK_INDEX_TYPE_UINT16);
                vkCmdDrawIndexedIndirect(cmd,buffer,indexed_offset,DRAWS,sizeof(VkDrawIndexedIndirectCommand));
            } else vkCmdDrawIndirect(cmd,buffer,indirect_offset,DRAWS,sizeof(VkDrawIndirectCommand));
        } else for (unsigned d=0;d<DRAWS;d++) {
            unsigned cell=indexed?(d*17+32)%DRAWS:d;
            assert(cell*stride*2<=UINT32_MAX);uint32_t offsets[2]={(uint32_t)(cell*stride*2),hybrid?0:(uint32_t)(cell*stride*2)};
            vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,layout,0,3,sets,2,offsets);
            if(indexed) {
                VkDeviceSize off=cell*stride*2;vkCmdBindVertexBuffers(cmd,0,1,&buffer,&off);vkCmdBindIndexBuffer(cmd,buffer,indices_offset+cell*16,VK_INDEX_TYPE_UINT16);
                vkCmdDrawIndexed(cmd,6,1,0,0,0);
            } else vkCmdDraw(cmd,6,1,0,0);
        }
        vkCmdEndRendering(cmd);
        image_barrier.srcAccessMask=VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;image_barrier.dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT;
        image_barrier.oldLayout=VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;image_barrier.newLayout=VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,0,NULL,0,NULL,1,&image_barrier);
        VkBufferImageCopy copy={.bufferOffset=output_offset,.imageSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1},.imageExtent={WIDTH,HEIGHT,1}};
        vkCmdCopyImageToBuffer(cmd,image,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,buffer,1,&copy);
        VkBufferMemoryBarrier readback={.sType=VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT,.dstAccessMask=VK_ACCESS_HOST_READ_BIT,
            .srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,.buffer=buffer,.offset=output_offset,.size=output_size};
        vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,0,NULL,1,&readback,0,NULL);
        CHECK(vkEndCommandBuffer(cmd));VkSubmitInfo submit={.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,.commandBufferCount=1,.pCommandBuffers=&cmd};CHECK(vkQueueSubmit(queue,1,&submit,VK_NULL_HANDLE));CHECK(vkQueueWaitIdle(queue));
        uint32_t *pixels=(void*)(mapped+output_offset);
        for(unsigned y=0;y<HEIGHT;y++)for(unsigned x=0;x<WIDTH;x++)for(unsigned c=0;c<4;c++) {
            unsigned d=hybrid?0:x/8+(y/8)*8;uint32_t got=pixels[(y*WIDTH+x)*4+c],expected=colors[d][c];
            if(got!=expected){fprintf(stderr,"variant %u pixel(%u,%u)[%u]: got %08x expected %08x\n",variant,x,y,c,got,expected);return 1;}
        }
        printf("%s, %s%s: all 4096 RGBA32_UINT pixels match independent CPU oracle on %s\n",native?"One native multi-draw / 64 draw records":"64 reference UBO-bound draws",indexed?"indexed geometry rebasing":"constant-derived geometry",hybrid?", original fragment UBO":"",properties.deviceName);
        vkDestroyPipeline(device,pipeline,NULL);vkDestroyDescriptorPool(device,descriptors,NULL);vkDestroyPipelineLayout(device,layout,NULL);
        for(unsigned i=0;i<3;i++)vkDestroyDescriptorSetLayout(device,layouts[i],NULL);
        for(unsigned i=0;i<2;i++)vkDestroyShaderModule(device,shaders[i],NULL);
    }
    shaderc_compile_options_release(options);shaderc_compiler_release(compiler);free(native_vs);free(native_fs);free(native_geometry_vs);
    vkDestroyCommandPool(device,pool,NULL);vkDestroyImageView(device,view,NULL);vkDestroyImage(device,image,NULL);vkFreeMemory(device,image_memory,NULL);
    vkUnmapMemory(device,memory);vkDestroyBuffer(device,buffer,NULL);vkFreeMemory(device,memory,NULL);vkDestroyDevice(device,NULL);vkDestroyInstance(instance,NULL);
    puts("Draw-ID propagation, nonzero table base and separate vertex/fragment constants verified; not a game performance result.");
    return 0;
}
