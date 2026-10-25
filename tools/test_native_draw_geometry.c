#include <assert.h>
#include <stdio.h>
#include <string.h>
#include "../runtime/nvn/vulkan/native_draw_geometry.h"
int main(void) {
    uint64_t a[17]={0},b[17]={0};uint32_t stride[17]={16,32};uint32_t divisor[17]={0};
    int32_t rebased;int64_t delta;
    a[0]=1600;a[1]=3200;b[0]=1760;b[1]=3520;
    assert(native_draw_vertex_rebase(3,a,b,stride,divisor,3,&rebased,&delta)&&rebased==13&&delta==10);
    assert(!native_draw_vertex_rebase(3,a,b,stride,divisor,-3,&rebased,&delta));
    assert(!native_draw_vertex_rebase(3,b,a,stride,divisor,3,&rebased,&delta));
    // The fetch address must match for every vertex/stream, even after rebasing.
    uint32_t rng=71;
    for(unsigned run=0;run<200000;run++) {
        rng=rng*1664525u+1013904223u;int move=(int)(rng%10001)-5000;
        for(unsigned s=0;s<2;s++)b[s]=a[s]+(int64_t)move*stride[s]+1600000;
        assert(native_draw_vertex_rebase(1,a,b,stride,divisor,0,&rebased,&delta));
        for(unsigned v=0;v<3;v++)assert(a[0]+((int64_t)v+rebased)*stride[0]==b[0]+(uint64_t)v*stride[0]);
        b[0]=a[0]+(int64_t)move*stride[0]+1600000;b[1]=a[1]+(int64_t)move*stride[1]+3200000;
        assert(native_draw_vertex_rebase(3,a,b,stride,divisor,0,&rebased,&delta));
        for(unsigned s=0;s<2;s++)for(unsigned v=0;v<3;v++)assert(a[s]+((int64_t)v+rebased)*stride[s]==b[s]+(uint64_t)v*stride[s]);
    }
    memcpy(b,a,sizeof a);b[0]++;assert(!native_draw_vertex_rebase(3,a,b,stride,divisor,0,&rebased,&delta));
    b[0]=a[0]+16;assert(!native_draw_vertex_rebase(3,a,b,stride,divisor,0,&rebased,&delta));
    b[1]=a[1]+32;divisor[1]=1;assert(!native_draw_vertex_rebase(3,a,b,stride,divisor,0,&rebased,&delta));
    divisor[1]=0;assert(!native_draw_vertex_rebase(3,a,b,stride,divisor,INT32_MAX,&rebased,&delta));
    a[0]=0;b[0]=UINT64_MAX;stride[0]=1;assert(!native_draw_vertex_rebase(1,a,b,stride,divisor,0,&rebased,&delta));
    uint64_t binding;uint32_t first_a,first_b;
    assert(native_draw_index_rebase(128,192,2,&binding,&first_a,&first_b)&&binding==128&&first_a==0&&first_b==32);
    assert(native_draw_index_rebase(192,128,4,&binding,&first_a,&first_b)&&binding==128&&first_a==16&&first_b==0);
    assert(!native_draw_index_rebase(1,128,2,&binding,&first_a,&first_b));
    assert(!native_draw_index_rebase(0,((uint64_t)UINT32_MAX+1)*4,4,&binding,&first_a,&first_b));
    puts("native draw geometry: 200000 address-equivalence trials; negative-rebase rejection, stride/divisor incompatibility and overflow guards passed");
    return 0;
}
