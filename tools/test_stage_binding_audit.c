#include <assert.h>
#include <stdio.h>
#include "../runtime/nvn/vulkan/stage_binding_audit.h"
static uint8_t bytes[128];
static const void *resolve(uint64_t buffer, uint64_t offset, uint64_t size) {
    return buffer == 1 && offset <= sizeof bytes && size <= sizeof bytes - offset ? bytes + offset : NULL;
}
int main(void) {
    StageBindingAudit a = {.n=3}, b;
    a.binding[0][0] = ((uint64_t)VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC << 32) | 7;
    a.binding[1][0] = ((uint64_t)VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC << 32) | 3;
    a.binding[2][0] = ((uint64_t)VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER << 32) | 32;
    a.binding[0][1]=a.binding[1][1]=1; a.binding[0][3]=a.binding[1][3]=16;
    b=a;
    uint32_t ad[2]={0,16}, bd[2]={32,48};
    for(int i=0;i<32;i++) bytes[i]=bytes[i+32]=(uint8_t)(i+1);
    assert(stage_audit_equal(&a,ad,2,&b,bd,2,resolve)); // descriptor order differs from dynamic order
    bytes[48]++;
    assert(!stage_audit_equal(&a,ad,2,&b,bd,2,resolve)); bytes[48]--;
    b.binding[2][2]=99; assert(!stage_audit_equal(&a,ad,2,&b,bd,2,resolve)); b=a;
    b.binding[0][1]=2; assert(!stage_audit_equal(&a,ad,2,&b,bd,2,resolve)); b=a;
    assert(!stage_audit_equal(&a,ad,1,&b,bd,1,resolve));
    assert(!stage_audit_equal(&a,ad,3,&b,bd,3,resolve));
    bd[1]=128; assert(!stage_audit_equal(&a,ad,2,&b,bd,2,resolve)); bd[1]=48;
    b.binding[0][3]=15; assert(!stage_audit_equal(&a,ad,2,&b,bd,2,resolve)); b=a;
    a.binding[0][0]=b.binding[0][0]=((uint64_t)VK_DESCRIPTOR_TYPE_STORAGE_BUFFER << 32) | 7;
    b.binding[0][2]=32; assert(!stage_audit_equal(&a,ad,1,&b,bd,1,resolve));
    puts("stage binding semantic audit tests passed");
}
