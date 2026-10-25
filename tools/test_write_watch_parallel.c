// Real Windows write-watch coverage and worker lifecycle/failure checks.
#include <assert.h>
#include <stdio.h>
#include <windows.h>
#include "../runtime/nvn/vulkan/geometry_watch.h"

static volatile LONG force_failure;
static uint64_t plat_now_ns(void) {
    LARGE_INTEGER t,f; QueryPerformanceCounter(&t); QueryPerformanceFrequency(&f);
    return (uint64_t)((unsigned __int128)t.QuadPart*1000000000u/f.QuadPart);
}
static size_t plat_peek_written_pages(uint64_t lo, uint64_t size, uint64_t *pages, size_t cap) {
    if (InterlockedCompareExchange(&force_failure,0,0)) return SIZE_MAX;
    ULONG_PTR n=cap; ULONG gran=0;
    if (GetWriteWatch(0,(void *)(uintptr_t)lo,size,(void **)pages,&n,&gran) || gran!=4096) return SIZE_MAX;
    return (size_t)n;
}
#include "../runtime/nvn/vulkan/write_watch_parallel.h"

int main(void) {
    volatile uint8_t *mem=VirtualAlloc(NULL,65536,MEM_RESERVE|MEM_COMMIT|MEM_WRITE_WATCH,PAGE_READWRITE);
    assert(mem);
    uint64_t base=(uintptr_t)mem, bytes=0;
    GeometryWatchSpan spans[]={{base,base+4*4096},{base+7*4096,base+16*4096}};
    for (unsigned workers=1;workers<=4;workers++) {
        assert(wwp_partition(spans,2,workers,&bytes) && bytes==13*4096);
        unsigned coverage[16]={0};
        for (unsigned w=0;w<workers;w++) for (unsigned r=0;r<wwp_job[w].nspans;r++) {
            GeometryWatchSpan s=wwp_job[w].spans[r];
            assert(s.lo>=base && s.hi<=base+65536 && !((s.lo|s.hi)&4095));
            for (uint64_t p=s.lo;p<s.hi;p+=4096) coverage[(p-base)>>12]++;
        }
        for (unsigned p=0;p<16;p++) assert(coverage[p]==(p<4 || p>=7));
    }
    GeometryWatchSpan bad[]={{base+1,base+4096}};
    assert(!wwp_partition(bad,1,2,&bytes));
    assert(!wwp_partition(spans,2,0,&bytes));
    assert(!wwp_partition(spans,2,5,&bytes));

    uint64_t result[33], serial[33], combined[33];
    // Grow the pool 1->4, then repeatedly change active count. New workers must
    // neither replay an old request nor miss their first request.
    for (unsigned round=0;round<160;round++) {
        unsigned workers=1+round%4;
        assert(!ResetWriteWatch((void *)mem,65536));
        for (unsigned p=0;p<16;p++) if ((p+round)%3) mem[p*4096]=(uint8_t)round;
        size_t a=0;
        for (unsigned i=0;i<2;i++) {
            size_t k=plat_peek_written_pages(spans[i].lo,spans[i].hi-spans[i].lo,serial+a,32-a);
            assert(k!=SIZE_MAX); a+=k;
        }
        result[32]=0x123456789abcdef0ull;
        WriteWatchStats stats;
        size_t n=wwp_query(spans,2,workers,result,32,&stats);
        assert(n==a && result[32]==0x123456789abcdef0ull && stats.bytes==13*4096);
        qsort(result,n,sizeof *result,geometry_watch_compare);
        qsort(serial,a,sizeof *serial,geometry_watch_compare);
        assert(!memcmp(result,serial,n*sizeof *result));
        // Neither serial nor parallel peek reset any flags, including gaps.
        size_t total=plat_peek_written_pages(base,65536,combined,32);
        size_t expected=0;
        for(unsigned p=0;p<16;p++) if((p+round)%3) expected++;
        assert(total==expected);
    }
    InterlockedExchange(&force_failure,1);
    assert(wwp_query(spans,2,3,result,32,NULL)==SIZE_MAX);
    InterlockedExchange(&force_failure,0);
    assert(wwp_query(spans,2,3,result,32,NULL)!=SIZE_MAX);
    result[1]=0x123456789abcdef0ull;
    assert(wwp_query(spans,2,3,result,1,NULL)==SIZE_MAX);
    assert(result[1]==0x123456789abcdef0ull);
    assert(wwp_query(NULL,0,4,result,32,NULL)==0);
    assert(VirtualFree((void *)mem,0,MEM_RELEASE));
    puts("Parallel write watch: exact disjoint coverage, 160 serial matches, pool growth, failure recovery, no reset or output overrun.");
    return 0;
}
