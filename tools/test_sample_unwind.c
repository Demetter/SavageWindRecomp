#include <assert.h>
#include <stdio.h>
#include <string.h>
#include "../runtime/platform/sample_unwind.h"

int main(void) {
    assert(sample_unwind_init());
    unsigned char *stack = VirtualAlloc(NULL, 3*4096, MEM_RESERVE, PAGE_NOACCESS);
    assert(stack && VirtualAlloc(stack, 4096, MEM_COMMIT, PAGE_READWRITE | PAGE_GUARD));
    assert(VirtualAlloc(stack+4096, 4096, MEM_COMMIT, PAGE_READWRITE));
    DWORD64 lo=(DWORD64)(uintptr_t)(stack+4096), hi=lo+4096;
    CONTEXT ctx={0};
    ctx.Rsp=hi-8; ctx.Rip=0x1234;
    *(DWORD64 *)(uintptr_t)ctx.Rsp=0x5678;
    assert(sample_unwind_frame(&ctx,0,NULL,lo,hi) && ctx.Rsp==hi && ctx.Rip==0x5678);
    ctx.Rsp=lo-8;
    assert(!sample_unwind_frame(&ctx,0,NULL,lo,hi));
    ctx.Rsp=hi;
    assert(!sample_unwind_frame(&ctx,0,NULL,lo,hi));

    // Valid unwind metadata for a 128-byte frame, but the supplied context is
    // only 16 bytes below StackBase. The old unbounded unwinder reads beyond it.
    unsigned char code[128];
    memset(code,0x90,sizeof code);
    code[64]=1; code[65]=4; code[66]=1; code[67]=0;
    code[68]=4; code[69]=0xf2; code[70]=code[71]=0;
    RUNTIME_FUNCTION fn={.BeginAddress=0,.EndAddress=16,.UnwindData=64};
    DWORD64 base=(DWORD64)(uintptr_t)code;
    ctx.Rsp=hi-16; ctx.Rip=base+8;
    assert(!sample_unwind_frame(&ctx,base,&fn,lo,hi));
    ctx.Rsp=lo+256; ctx.Rip=base+8;
    *(DWORD64 *)(uintptr_t)(ctx.Rsp+128)=0x8765;
    assert(sample_unwind_frame(&ctx,base,&fn,lo,hi));
    assert(ctx.Rsp==lo+392 && ctx.Rip==0x8765);
    MEMORY_BASIC_INFORMATION info;
    assert(VirtualQuery(stack,&info,sizeof info)==sizeof info && (info.Protect & PAGE_GUARD));
    assert(VirtualQuery(stack+8192,&info,sizeof info)==sizeof info && info.State==MEM_RESERVE);
    VirtualFree(stack,0,MEM_RELEASE);
    puts("PASS: leaf and framed unwind; lower guard and upper reserved boundary rejected without touching either");
    return 0;
}
