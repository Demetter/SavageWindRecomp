// Isolated native-ABI sizing only. Capture guest pages lazily from a debugger-
// stopped game; writes affect this process's private replay, never the game.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <x86intrin.h>
#include "entries.h"

volatile int g_fnprof;
void fnprof_hit(uint64_t a,uint64_t b,uint64_t d) {(void)a;(void)b;(void)d; abort();}
void rt_udf(Cpu *c,uint64_t pc) {(void)c; fprintf(stderr,"unexpected UDF %llx\n",(unsigned long long)pc);exit(91);}
typedef struct { uint64_t addr; unsigned char initial[4096], expected[4096]; } Page;
static Page *pages;
static size_t page_count, capacity;
static HANDLE source_process;
static int phase;
static uint64_t sink;
#ifdef ABI_USE_DLL
GuestFn abi_lookup(uint64_t);
void abi_counts_reset(void),abi_counts_print(void);
#endif

static Page *map_page(uint64_t a) {
    a &= ~4095ull;
    for(size_t i=0;i<page_count;i++) if(pages[i].addr==a)return &pages[i];
    if(page_count==capacity) {capacity=capacity?capacity*2:128;pages=realloc(pages,capacity*sizeof *pages); if(!pages)abort();}
    Page *p=&pages[page_count];p->addr=a;
    // Reserving an already reserved 64-KiB unit fails harmlessly; commit the
    // individual page. All addresses must remain inside the game's window.
    VirtualAlloc((void*)(uintptr_t)(a&~65535ull),65536,MEM_RESERVE,PAGE_NOACCESS);
    if(!VirtualAlloc((void*)(uintptr_t)a,4096,MEM_COMMIT,PAGE_READWRITE)) {fprintf(stderr,"map fail %llx %lu\n",(unsigned long long)a,GetLastError());exit(92);}
    page_count++;return p;
}
static LONG WINAPI replay_fault(EXCEPTION_POINTERS *e) {
    if(e->ExceptionRecord->ExceptionCode!=EXCEPTION_ACCESS_VIOLATION || !source_process)return EXCEPTION_CONTINUE_SEARCH;
    uint64_t a=e->ExceptionRecord->ExceptionInformation[1]&~4095ull;
    if(a<0x7000000000ull || a>=0x7800000000ull) {fprintf(stderr,"outside guest: %llx phase=%d\n",(unsigned long long)a,phase);exit(93);}
    size_t before=page_count;Page *p=map_page(a);
    if(page_count==before) {fprintf(stderr,"repeat fault %llx\n",(unsigned long long)a);exit(94);}
    SIZE_T got=0;
    if(!ReadProcessMemory(source_process,(void*)(uintptr_t)a,p->initial,4096,&got)||got!=4096) {fprintf(stderr,"read fail %llx %lu\n",(unsigned long long)a,GetLastError());exit(95);}
    memcpy((void*)(uintptr_t)a,p->initial,4096);memcpy(p->expected,p->initial,4096);
    return EXCEPTION_CONTINUE_EXECUTION;
}
static void restore(void) {for(size_t i=0;i<page_count;i++)memcpy((void*)(uintptr_t)pages[i].addr,pages[i].initial,4096);}
static const ProbeEntry *entry(const char *name) {
    for(size_t i=0;i<sizeof entries/sizeof *entries;i++)if(!strcmp(entries[i].name,name)) {
#ifdef ABI_USE_DLL
        static ProbeEntry e;e=entries[i];e.old=abi_lookup(strtoull(name,NULL,16));return &e;
#else
        return &entries[i];
#endif
    }
    fprintf(stderr,"unknown root %s\n",name);exit(96);
}
static void read_exact(FILE *f,void *p,size_t n) {if(fread(p,1,n,f)!=n){fprintf(stderr,"short input\n");exit(97);}}
static int compare(Cpu *a,Cpu *b) {
    int bad=0;
    if(memcmp(a,b,sizeof *a)) {
        size_t i=0;while(i<sizeof *a && ((unsigned char*)a)[i]==((unsigned char*)b)[i])i++;
        fprintf(stderr,"CPU mismatch byte %zu old=%02x native=%02x\n",i,((unsigned char*)a)[i],((unsigned char*)b)[i]);bad=1;
    }
    for(size_t i=0;i<page_count;i++)if(memcmp((void*)(uintptr_t)pages[i].addr,pages[i].expected,4096)) {
        size_t k=0;while(k<4096 && ((unsigned char*)(uintptr_t)pages[i].addr)[k]==pages[i].expected[k])k++;
        fprintf(stderr,"memory mismatch %llx + %zu\n",(unsigned long long)pages[i].addr,k);bad=1;break;
    }
    return bad;
}
static int validate(const ProbeEntry *p,Cpu seed) {
#ifdef ABI_USE_DLL
    abi_counts_reset();
#endif
    Cpu a=seed,b=seed;restore();phase=0;p->old(&a);
    for(size_t i=0;i<page_count;i++)memcpy(pages[i].expected,(void*)(uintptr_t)pages[i].addr,4096);
    restore();phase=1;p->native(&b);
#ifdef ABI_USE_DLL
    abi_counts_print();
#endif
    return compare(&a,&b);
}
static uint64_t ticks(void) {_mm_lfence();return __rdtsc();}
static uint64_t ticks_end(void) {unsigned aux;uint64_t v=__rdtscp(&aux);_mm_lfence();return v;}
static double tsc_hz(void) {
    LARGE_INTEGER f,a,b;QueryPerformanceFrequency(&f);QueryPerformanceCounter(&a);uint64_t t=ticks();
    Sleep(100);uint64_t u=ticks_end();QueryPerformanceCounter(&b);return (double)(u-t)*f.QuadPart/(b.QuadPart-a.QuadPart);
}
int main(int argc,char **argv) {
    if(argc<3)return 2;
    AddVectoredExceptionHandler(1,replay_fault);
    Cpu seed;const ProbeEntry *p;char name[32]={0};
    if(!strcmp(argv[1],"capture")) {
        if(argc!=6)return 2;
        source_process=OpenProcess(PROCESS_VM_READ|PROCESS_QUERY_INFORMATION,FALSE,(DWORD)strtoul(argv[2],NULL,10));
        if(!source_process){fprintf(stderr,"open source failed %lu\n",GetLastError());return 3;}
        p=entry(argv[3]);FILE *f=fopen(argv[4],"rb");if(!f)return 4;read_exact(f,&seed,sizeof seed);fclose(f);
        int bad=validate(p,seed);
        f=fopen(argv[5],"wb");if(!f)return 5;
        uint64_t magic=0x41424950524f4245ull,n=page_count;
        snprintf(name,sizeof name,"%s",p->name);fwrite(&magic,8,1,f);fwrite(name,1,32,f);fwrite(&seed,sizeof seed,1,f);fwrite(&n,8,1,f);
        for(size_t i=0;i<page_count;i++){fwrite(&pages[i].addr,8,1,f);fwrite(pages[i].initial,1,4096,f);}fclose(f);
        printf("{\"root\":\"%s\",\"pages\":%zu,\"cpu_bytes\":%zu,\"match\":%s}\n",name,page_count,sizeof(Cpu),bad?"false":"true");return bad?10:0;
    }
    FILE *f=fopen(argv[2],"rb");if(!f)return 4;uint64_t magic,n;read_exact(f,&magic,8);if(magic!=0x41424950524f4245ull)return 5;
    read_exact(f,name,32);read_exact(f,&seed,sizeof seed);read_exact(f,&n,8);
    for(size_t i=0;i<n;i++){uint64_t a;read_exact(f,&a,8);Page *pg=map_page(a);read_exact(f,pg->initial,4096);}fclose(f);p=entry(name);
    if(validate(p,seed))return 10;
    if(!strcmp(argv[1],"verify")){printf("verified %s pages=%zu\n",name,page_count);return 0;}
    // An isolated replay process only. Pinning prevents cross-core TSC effects;
    // no affinity or priority changes are applied to the game/user's programs.
    SetThreadAffinityMask(GetCurrentThread(),(DWORD_PTR)1<<4);
    double hz=tsc_hz();const int iterations=2048,blocks=16;
    printf("root,block,mode,iterations,total_cycles,ns_per_call,reset_excluded\n");
    for(int block=-4;block<blocks;block++) {
        int mode=(block+4)%4==1||(block+4)%4==2;GuestFn fn=mode?p->native:p->old;
        uint64_t total=0;
        for(int i=0;i<iterations;i++) {
            restore();Cpu c=seed;uint64_t a=ticks();fn(&c);uint64_t b=ticks_end();total+=b-a;sink+=c.x[0];
        }
        if(block>=0)printf("%s,%d,%d,%d,%llu,%.6f,1\n",name,block,mode,iterations,(unsigned long long)total,1e9*total/hz/iterations);
    }
    return sink==0xdeadbeef12345678ull;
}
