// GPU equivalence test for the native per-draw constant table building block.
// Reference: 64 dispatches with dynamic UBO bindings. Candidate: one dispatch
// selects 64 different records, with bounded raw addresses and no CB descriptors.
#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>
#include <shaderc/shaderc.h>
#include "../runtime/nvn/vulkan/native_draw_constants.h"
#define CHECK(x) do { VkResult r_=(x); if(r_!=VK_SUCCESS){fprintf(stderr,"%s: %d\n",#x,r_);exit(1);} }while(0)
enum { DRAWS=64, WORDS=40 };
_Static_assert(sizeof(NativeDrawConstant)==16 && sizeof(NativeDrawConstants)==576, "GLSL std430 record layout");

int main(void) {
    const char *source =
        "#version 460\nlayout(local_size_x=1) in;\n"
        "layout(push_constant) uniform Index {uint draw;} index_;\n"
        "uint draw_index() {return index_.draw;}\n"
        "#define cbr(cb,off) cb[min((off)>>4,uint(cb.length()-1))][((off)>>2)&3u]\n"
        "layout(std140, set = 0, binding = 0) uniform CB0 { uvec4 cb0[4]; };\n"
        "layout(std140, set = 0, binding = 3) uniform CB3 { uvec4 cb3[8]; };\n"
        "layout(std430,set=0,binding=17) buffer Out {uint v[];} result_;\n"
        "void main() {uint d=draw_index(); for(uint i=0u;i<32u;i++) result_.v[d*40u+i]=cbr(cb3,i*4u);\n"
        " result_.v[d*40u+32u]=cbr(cb3,0xffffffffu); result_.v[d*40u+33u]=cbr(cb3,128u);\n"
        " result_.v[d*40u+34u]=cb0[0][0]; result_.v[d*40u+35u]=cb0[1][3];\n"
        " result_.v[d*40u+36u]=cb3[7][0]; result_.v[d*40u+37u]=cb3[7][3];\n"
        " result_.v[d*40u+38u]=floatBitsToUint(fma(uintBitsToFloat(cb3[0][0]),uintBitsToFloat(cb3[0][1]),uintBitsToFloat(cb3[0][2])));\n"
        " result_.v[d*40u+39u]=cb3[2][1];}\n";
    uint32_t mask;
    char *native = vk_shader_native_draw_constants(source,0,"draw_index()",&mask);
    assert(native && mask==9 && !strstr(native,"uniform CB"));
    // Keep draw_index before the generated accessors. Both indexing sites use
    // this function, including the output, so each invocation proves its own row.
    const char *old="return index_.draw;", *replacement="return gl_GlobalInvocationID.x;";
    char *at=strstr(native,old); assert(at);
    size_t delta=strlen(replacement)-strlen(old), len=strlen(native), pos=(size_t)(at-native);
    native=realloc(native,len+delta+1); assert(native); at=native+pos;
    memmove(at+strlen(replacement),at+strlen(old),len-pos-strlen(old)+1);
    memcpy(at,replacement,strlen(replacement));

    VkApplicationInfo app={.sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,.apiVersion=VK_API_VERSION_1_3};
    VkInstanceCreateInfo ici={.sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,.pApplicationInfo=&app};
    VkInstance instance; CHECK(vkCreateInstance(&ici,NULL,&instance));
    VkPhysicalDevice devices[16], physical=VK_NULL_HANDLE; uint32_t nd=16,family=0;
    CHECK(vkEnumeratePhysicalDevices(instance,&nd,devices));
    for(unsigned d=0;d<nd&&!physical;d++){
        VkQueueFamilyProperties p[32]; uint32_t n=32; vkGetPhysicalDeviceQueueFamilyProperties(devices[d],&n,p);
        for(unsigned q=0;q<n;q++) if(p[q].queueFlags&VK_QUEUE_COMPUTE_BIT){physical=devices[d];family=q;break;}
    }
    assert(physical); VkPhysicalDeviceProperties properties; vkGetPhysicalDeviceProperties(physical,&properties);
    VkPhysicalDeviceVulkan12Features have12={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceFeatures2 have={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,.pNext=&have12};
    vkGetPhysicalDeviceFeatures2(physical,&have); assert(have12.bufferDeviceAddress&&have.features.shaderInt64);
    float priority=1; VkDeviceQueueCreateInfo qci={.sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,.queueFamilyIndex=family,.queueCount=1,.pQueuePriorities=&priority};
    VkPhysicalDeviceVulkan12Features f12={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,.bufferDeviceAddress=VK_TRUE};
    VkPhysicalDeviceFeatures features={.shaderInt64=VK_TRUE,.robustBufferAccess=have.features.robustBufferAccess};
    VkDeviceCreateInfo dci={.sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,.pNext=&f12,.pEnabledFeatures=&features,.queueCreateInfoCount=1,.pQueueCreateInfos=&qci};
    VkDevice device; CHECK(vkCreateDevice(physical,&dci,NULL,&device)); VkQueue queue; vkGetDeviceQueue(device,family,0,&queue);
    VkDeviceSize stride=properties.limits.minUniformBufferOffsetAlignment; if(stride<256)stride=256;
    VkDeviceSize native_offset=DRAWS*stride,table_offset=2*native_offset;
    VkDeviceSize alignment=properties.limits.minStorageBufferOffsetAlignment;
    VkDeviceSize output_offset=(table_offset+sizeof(NativeDrawConstants)*DRAWS+alignment-1)/alignment*alignment;
    VkDeviceSize total=output_offset+DRAWS*WORDS*4;
    VkBufferCreateInfo bci={.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,.size=total,.usage=VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT|VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT};
    VkBuffer buffer; CHECK(vkCreateBuffer(device,&bci,NULL,&buffer)); VkMemoryRequirements req; vkGetBufferMemoryRequirements(device,buffer,&req);
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(physical,&mp); unsigned type=0;
    for(;type<mp.memoryTypeCount;type++)if((req.memoryTypeBits&(1u<<type))&&(mp.memoryTypes[type].propertyFlags&(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))==(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))break;
    assert(type<mp.memoryTypeCount);
    VkMemoryAllocateFlagsInfo flags={.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,.flags=VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT};
    VkMemoryAllocateInfo mai={.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,.pNext=&flags,.allocationSize=req.size,.memoryTypeIndex=type};
    VkDeviceMemory memory; CHECK(vkAllocateMemory(device,&mai,NULL,&memory)); CHECK(vkBindBufferMemory(device,buffer,memory,0));
    uint8_t *mapped; CHECK(vkMapMemory(device,memory,0,req.size,0,(void**)&mapped)); memset(mapped,0,(size_t)total);
    VkBufferDeviceAddressInfo address_info={.sType=VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,.buffer=buffer};
    VkDeviceAddress address=vkGetBufferDeviceAddress(device,&address_info); assert(address&&address%4==0);
    NativeDrawConstants *table=(void*)(mapped+table_offset);
    uint32_t expected[DRAWS][WORDS];
    for(unsigned d=0;d<DRAWS;d++){
        uint32_t *ref=(void*)(mapped+d*stride),*raw=(void*)(mapped+native_offset+d*stride);
        unsigned words=d%17==0?0:d%3==0?29:d%3==1?31:32;
        for(unsigned i=0;i<32;i++)raw[i]=(d*0x9e3779b9u)^(i*0x85ebca6bu)^0x7fc12345u;
        raw[0]=0x3fa00000u; raw[1]=0xc0000000u; raw[2]=0x3f000000u;
        raw[3]=0x80000000u;raw[4]=0x7fc12345u;raw[5]=0xffffffffu;
        memcpy(ref,raw,words*4); memcpy(expected[d],ref,128);
        expected[d][32]=ref[31];expected[d][33]=ref[28];expected[d][34]=ref[0];expected[d][35]=ref[7];
        expected[d][36]=ref[28];expected[d][37]=ref[31];expected[d][38]=words?0xc0000000u:0;expected[d][39]=ref[9];
        table[d].constants[0]=(NativeDrawConstant){words?address+native_offset+d*stride:0,words,0};
        table[d].constants[3]=table[d].constants[0];
    }
    shaderc_compiler_t compiler=shaderc_compiler_initialize(); shaderc_compile_options_t options=shaderc_compile_options_initialize();
    shaderc_compile_options_set_target_env(options,shaderc_target_env_vulkan,shaderc_env_version_vulkan_1_2);
    shaderc_compile_options_set_optimization_level(options,shaderc_optimization_level_performance);
    VkCommandPoolCreateInfo pci={.sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,.queueFamilyIndex=family,.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT};
    VkCommandPool pool; CHECK(vkCreateCommandPool(device,&pci,NULL,&pool));
    VkCommandBufferAllocateInfo cai={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,.commandPool=pool,.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,.commandBufferCount=1};
    VkCommandBuffer cmd; CHECK(vkAllocateCommandBuffers(device,&cai,&cmd));
    for(unsigned variant=0;variant<2;variant++){
        const char *text=variant?native:source;
        shaderc_compilation_result_t compiled=shaderc_compile_into_spv(compiler,text,strlen(text),shaderc_compute_shader,"native-draw-constants","main",options);
        if(shaderc_result_get_compilation_status(compiled)!=shaderc_compilation_status_success){fprintf(stderr,"%s\n",shaderc_result_get_error_message(compiled));return 1;}
        VkShaderModuleCreateInfo smci={.sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,.codeSize=shaderc_result_get_length(compiled),.pCode=(const uint32_t*)shaderc_result_get_bytes(compiled)};
        VkShaderModule module; CHECK(vkCreateShaderModule(device,&smci,NULL,&module));
        VkDescriptorSetLayoutBinding bindings[3]={{17,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,NULL},{0,VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,1,VK_SHADER_STAGE_COMPUTE_BIT,NULL},{3,VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,1,VK_SHADER_STAGE_COMPUTE_BIT,NULL}};
        VkDescriptorSetLayoutBinding table_binding={0,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,NULL};
        VkDescriptorSetLayout layouts[3];
        for(unsigned s=0;s<3;s++){
            VkDescriptorSetLayoutCreateInfo slci={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,.bindingCount=s==0?(variant?1:3):s==2?1:0,.pBindings=s==0?bindings:&table_binding};
            CHECK(vkCreateDescriptorSetLayout(device,&slci,NULL,&layouts[s]));
        }
        VkPushConstantRange push={VK_SHADER_STAGE_COMPUTE_BIT,0,4};
        VkPipelineLayoutCreateInfo plci={.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,.setLayoutCount=3,.pSetLayouts=layouts,.pushConstantRangeCount=1,.pPushConstantRanges=&push};
        VkPipelineLayout layout; CHECK(vkCreatePipelineLayout(device,&plci,NULL,&layout));
        VkComputePipelineCreateInfo cpci={.sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,.layout=layout,.stage={.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,.stage=VK_SHADER_STAGE_COMPUTE_BIT,.module=module,.pName="main"}};
        VkPipeline pipeline; CHECK(vkCreateComputePipelines(device,VK_NULL_HANDLE,1,&cpci,NULL,&pipeline));
        VkDescriptorPoolSize sizes[2]={{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,2},{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,2}};
        VkDescriptorPoolCreateInfo dpci={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,.maxSets=3,.poolSizeCount=2,.pPoolSizes=sizes};
        VkDescriptorPool dp; CHECK(vkCreateDescriptorPool(device,&dpci,NULL,&dp));
        VkDescriptorSetAllocateInfo dsai={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,.descriptorPool=dp,.descriptorSetCount=3,.pSetLayouts=layouts};
        VkDescriptorSet sets[3]; CHECK(vkAllocateDescriptorSets(device,&dsai,sets));
        VkDescriptorBufferInfo bi[4]={{buffer,output_offset,DRAWS*WORDS*4},{buffer,0,128},{buffer,0,128},{buffer,table_offset,DRAWS*sizeof(NativeDrawConstants)}};
        VkWriteDescriptorSet writes[4];
        for(unsigned i=0;i<4;i++)writes[i]=(VkWriteDescriptorSet){.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,.dstSet=i==3?sets[2]:sets[0],.dstBinding=i==3?0:bindings[i].binding,.descriptorCount=1,.descriptorType=i==3?VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:bindings[i].descriptorType,.pBufferInfo=&bi[i]};
        if(variant){vkUpdateDescriptorSets(device,1,writes,0,NULL);vkUpdateDescriptorSets(device,1,writes+3,0,NULL);}else vkUpdateDescriptorSets(device,4,writes,0,NULL);
        memset(mapped+output_offset,0xa5,DRAWS*WORDS*4); CHECK(vkResetCommandBuffer(cmd,0));
        VkCommandBufferBeginInfo begin={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};CHECK(vkBeginCommandBuffer(cmd,&begin));
        vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,pipeline);
        if(variant){
            vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,layout,0,3,sets,0,NULL);
            vkCmdDispatch(cmd,DRAWS,1,1);
        }else for(unsigned d=0;d<DRAWS;d++){
            uint32_t offsets[2]={(uint32_t)(d*stride),(uint32_t)(d*stride)};
            vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,layout,0,3,sets,2,offsets);
            vkCmdPushConstants(cmd,layout,VK_SHADER_STAGE_COMPUTE_BIT,0,4,&d);vkCmdDispatch(cmd,1,1,1);
        }
        VkMemoryBarrier barrier={.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER,.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT,.dstAccessMask=VK_ACCESS_HOST_READ_BIT};
        vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,1,&barrier,0,NULL,0,NULL);
        CHECK(vkEndCommandBuffer(cmd));VkSubmitInfo submit={.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,.commandBufferCount=1,.pCommandBuffers=&cmd};
        CHECK(vkQueueSubmit(queue,1,&submit,VK_NULL_HANDLE));CHECK(vkQueueWaitIdle(queue));
        const uint32_t *got=(const void*)(mapped+output_offset);
        for(unsigned d=0;d<DRAWS;d++)for(unsigned i=0;i<WORDS;i++)if(got[d*WORDS+i]!=expected[d][i]){
            fprintf(stderr,"variant %u draw %u word %u: %08x != %08x\n",variant,d,i,got[d*WORDS+i],expected[d][i]);return 1;
        }
        vkDestroyDescriptorPool(device,dp,NULL);vkDestroyPipeline(device,pipeline,NULL);vkDestroyPipelineLayout(device,layout,NULL);
        for(unsigned s=0;s<3;s++)vkDestroyDescriptorSetLayout(device,layouts[s],NULL);
        vkDestroyShaderModule(device,module,NULL);shaderc_result_release(compiled);
    }
    printf("%s: 64 distinct draw records, 2560 exact words per variant; static/dynamic/clamped loads, NaN/negative-zero payloads, partial vec4 bounds, null records and finite FMA passed.\n",properties.deviceName);
    shaderc_compile_options_release(options);shaderc_compiler_release(compiler);vkDestroyCommandPool(device,pool,NULL);
    vkUnmapMemory(device,memory);vkDestroyBuffer(device,buffer,NULL);vkFreeMemory(device,memory,NULL);
    vkDestroyDevice(device,NULL);vkDestroyInstance(instance,NULL);free(native);return 0;
}
