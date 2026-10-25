#include "gpu_consumer_probe.h"
#include "kernel.h"
#include <stdlib.h>
#include <stdatomic.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>

#define GCP_N 16384
#define GCP_W 16
typedef struct { uint64_t id, seq, frame, shader, addr, size, payload, applied; unsigned kind; uint64_t expected,stride; uint32_t row,rows; bool verify; } Result;
typedef struct { Result r; uint64_t lo, hi, arm, fence; bool active, gpu; } Watch;
static SRWLOCK lock = SRWLOCK_INIT;
static Result results[GCP_N];
static unsigned nr;
static Watch watches[GCP_W];
static uint64_t next_id, fence_id, pick;
static int mode;
static bool verify_readbacks;
static uint64_t epoch, start_ns=30000000000ull, duration_ns=45000000000ull;
static pthread_once_t once = PTHREAD_ONCE_INIT;
static void init(void) {
    const char *e=getenv("SAVAGEWIND_CONSUMER_PROBE"); mode=e?atoi(e):0;
    e=getenv("SAVAGEWIND_CONSUMER_VERIFY");verify_readbacks=e&&atoi(e)!=0;
    epoch=plat_now_ns();
    e=getenv("SAVAGEWIND_CONSUMER_START"); if(e)start_ns=(uint64_t)(atof(e)*1e9);
    if(mode) LOG("[gcp-config] mode %d start_ns %llu duration_ns %llu; diagnostic only; binding records are potential GPU access, not hardware faults\n",mode,(unsigned long long)start_ns,(unsigned long long)duration_ns);
}
bool gcp_active(void) { pthread_once(&once,init); uint64_t t=mode?plat_now_ns()-epoch:0; return mode && t>=start_ns && t<start_ns+duration_ns; }
static uint64_t row_hash(const void *p,uint32_t row,uint32_t rows,uint64_t stride) {
    uint64_t h=14695981039346656037ull;const uint8_t *b=p;
    for(uint32_t y=0;y<rows;y++)for(uint32_t x=0;x<row;x++)h=(h^b[y*stride+x])*1099511628211ull;
    return h;
}
void gcp_readback_expected(uint64_t seq,uint64_t addr,const void *gpu_rows,uint32_t row,uint32_t rows,uint64_t stride) {
    if(!gcp_active()||!verify_readbacks||!row||!rows||(uint64_t)row*rows>4096||stride*(rows-1)+row>4096)return;
    uint64_t expected=row_hash(gpu_rows,row,rows,row);
    AcquireSRWLockExclusive(&lock);
    for(unsigned i=0;i<nr;i++){Result *r=&results[i];if(r->kind==GCP_READBACK&&r->seq==seq&&r->addr==addr&&!r->applied){
        r->expected=expected;r->row=row;r->rows=rows;r->stride=stride;r->verify=true;break;}}
    ReleaseSRWLockExclusive(&lock);
}
static void open_watch(Watch *w) {
    if(!w->active)return;
    DWORD old; if(!VirtualProtect((void *)(uintptr_t)w->lo,w->hi-w->lo,PAGE_READWRITE,&old))
        FATAL("consumer probe restore failed: %lu\n",GetLastError());
    w->active=false;
}
void gcp_result(unsigned kind,uint64_t seq,uint64_t frame,uint64_t shader,uint64_t addr,uint64_t size,uint64_t payload) {
    if(!gcp_active())return;
    AcquireSRWLockExclusive(&lock);
    // Deduplicate identical ranges within a submission; size is declared extent,
    // not a claim that every byte is actually read/written by the shader.
    for(unsigned i=nr;i>0;i--) { Result *r=&results[i-1]; if(r->seq<seq)break;
        if(kind>GCP_QUERY&&r->seq==seq&&r->kind==kind&&r->shader==shader&&r->addr==addr&&r->size==size){ReleaseSRWLockExclusive(&lock);return;}}
    if(nr<GCP_N) results[nr++]=(Result){++next_id,seq,frame,shader,addr,size,payload,0,kind};
    else TRACE_LOG("[gcp-overflow] seq %llu\n",(unsigned long long)seq);
    ReleaseSRWLockExclusive(&lock);
}
void gcp_applied(unsigned kind,uint64_t seq,uint64_t addr,uint64_t hash) {
    if(!gcp_active())return;
    AcquireSRWLockExclusive(&lock);
    for(unsigned i=0;i<nr;i++) {Result *r=&results[i]; if(r->kind==kind&&r->seq==seq&&r->addr==addr&&!r->applied){
        r->applied=plat_now_ns();
        TRACE_LOG("[gcp-apply] %llu id %llu kind %u seq %llu addr %llx size %llu hash %llx\n",(unsigned long long)r->applied,(unsigned long long)r->id,kind,(unsigned long long)seq,(unsigned long long)addr,(unsigned long long)r->payload,(unsigned long long)hash);
        break;
    }}
    ReleaseSRWLockExclusive(&lock);
}
void gcp_boundary(uint64_t last,uint64_t done) {
    if(!gcp_active())return;
    AcquireSRWLockExclusive(&lock);
    TRACE_LOG("[gcp-boundary] %llu fence %llu last %llu done %llu records %u\n",(unsigned long long)plat_now_ns(),(unsigned long long)fence_id,(unsigned long long)last,(unsigned long long)done,nr);
    // Census once in 60 frames; all output records remain available for traps.
    if(fence_id%60==0)for(unsigned i=0;i<nr;i++){Result *r=&results[i];
        TRACE_LOG("[gcp-result] %llu id %llu kind %u seq %llu frame %llu shader %llx addr %llx extent %llu payload %llu pending %d applied %llu\n",(unsigned long long)plat_now_ns(),(unsigned long long)r->id,r->kind,(unsigned long long)r->seq,(unsigned long long)r->frame,(unsigned long long)r->shader,(unsigned long long)r->addr,(unsigned long long)r->size,(unsigned long long)r->payload,r->seq>done||((r->kind==GCP_READBACK||r->kind==GCP_QUERY)&&!r->applied),(unsigned long long)r->applied);}
    ReleaseSRWLockExclusive(&lock);
}
void gcp_gpu_use(const char *kind,uint64_t seq,uint64_t shader,uint64_t addr,uint64_t size) {
    if(!gcp_active()||!size)return;
    AcquireSRWLockExclusive(&lock);
    if(fence_id%60==0)for(unsigned i=0;i<nr;i++){Result *r=&results[i];
        if(r->kind!=GCP_GMEM_WRITE_BOUND||r->seq>seq||addr>=r->addr+r->size||r->addr>=addr+size)continue;
        TRACE_LOG("[gcp-link] %llu id %llu fence %llu producer_seq %llu consumer_seq %llu kind %s shader %llx addr %llx size %llu\n",(unsigned long long)plat_now_ns(),(unsigned long long)r->id,(unsigned long long)fence_id,(unsigned long long)r->seq,(unsigned long long)seq,kind,(unsigned long long)shader,(unsigned long long)addr,(unsigned long long)size);
    }
    for(unsigned i=0;i<GCP_W;i++){Watch *w=&watches[i];if(!w->active||w->gpu||addr>=w->r.addr+w->r.size||w->r.addr>=addr+size)continue;
        w->gpu=true;
        TRACE_LOG("[gcp-gpu-binding] %llu id %llu fence %llu producer_seq %llu consumer_seq %llu kind %s shader %llx addr %llx size %llu\n",(unsigned long long)plat_now_ns(),(unsigned long long)w->r.id,(unsigned long long)w->fence,(unsigned long long)w->r.seq,(unsigned long long)seq,kind,(unsigned long long)shader,(unsigned long long)addr,(unsigned long long)size);
    }
    ReleaseSRWLockExclusive(&lock);
}
void gcp_return(uint64_t seq,uint64_t start,uint64_t queued,uint64_t gpu_start,uint64_t gpu_end) {
    if(!gcp_active())return;
    AcquireSRWLockExclusive(&lock);
    TRACE_LOG("[gcp-return] %llu frame %llu seq %llu start %llu queued %llu gpu_start %llu gpu_end %llu\n",(unsigned long long)plat_now_ns(),(unsigned long long)fence_id,(unsigned long long)seq,(unsigned long long)start,(unsigned long long)queued,(unsigned long long)gpu_start,(unsigned long long)gpu_end);
    ReleaseSRWLockExclusive(&lock);
}
void gcp_layout(uint64_t addr,uint64_t row,uint64_t rows,uint64_t stride,uint64_t shader) {
    if(!gcp_active())return;
    static uint64_t seen[128];static unsigned n;
    AcquireSRWLockExclusive(&lock);
    for(unsigned i=0;i<n;i++)if(seen[i]==addr){ReleaseSRWLockExclusive(&lock);return;}
    if(n<128)seen[n++]=addr;
    TRACE_LOG("[gcp-layout] addr %llx row %llu rows %llu stride %llu shader %llx\n",(unsigned long long)addr,(unsigned long long)row,(unsigned long long)rows,(unsigned long long)stride,(unsigned long long)shader);
    ReleaseSRWLockExclusive(&lock);
}
void gcp_fence(bool end,uint64_t seq) {
    pthread_once(&once,init);if(!mode)return;
    bool collecting=gcp_active();
    AcquireSRWLockExclusive(&lock);
    if(!end){fence_id++;for(unsigned i=0;i<GCP_W;i++){Watch *w=&watches[i];if(w->active&&(!collecting||fence_id>=w->fence+3)){
        TRACE_LOG("[gcp-expire] %llu id %llu fence %llu gpu_binding %d\n",(unsigned long long)plat_now_ns(),(unsigned long long)w->r.id,(unsigned long long)fence_id,w->gpu);open_watch(w);}}}
    if(collecting)TRACE_LOG("[gcp-fence] %llu frame %llu end %d seq %llu\n",(unsigned long long)plat_now_ns(),(unsigned long long)fence_id,end,(unsigned long long)seq);
    if(end){
        // Sample after normal fence completion: never protect unfinished GPU
        // output, never wait from the fault handler. One rotating target/kind.
        if(collecting&&(mode==3?fence_id%120==30:fence_id%30==0)){
            unsigned choices[GCP_N],nc=0;
            for(unsigned i=0;i<nr;i++){Result *r=&results[i];bool output=r->kind==GCP_READBACK||r->kind==GCP_QUERY;
                if((!verify_readbacks||r->verify)&&r->seq<=seq&&r->size&&r->size<=4*1024*1024&&r->addr>=HEAP_BASE&&r->addr+r->size<=HEAP_BASE+HEAP_MAX&&
                    (mode==2?!output:output)&&(output?r->applied!=0:true))choices[nc++]=i;}
            // Up to four nonoverlapping ranges; rotate through candidates.
            for(unsigned j=0;j<nc&&j<(mode==3?1:4);j++){
                Result *r=&results[choices[(pick+j)%nc]];uint64_t lo=r->addr&~4095ull,hi=(r->addr+r->size+4095)&~4095ull;bool overlap=false;
                // Mode 3 bounds perturbation: one first page of one output.
                // A miss cannot establish that the rest of the range is unused.
                if(mode==3&&hi>lo+4096)hi=lo+4096;
                for(unsigned w=0;w<GCP_W;w++)overlap|=watches[w].active&&lo<watches[w].hi&&watches[w].lo<hi;
                if(overlap)continue;
                if(r->verify && row_hash((const void *)(uintptr_t)r->addr,r->row,r->rows,r->stride)!=r->expected){
                    TRACE_LOG("[gcp-verify-skip] id %llu seq %llu reason changed_before_arm\n",(unsigned long long)r->id,(unsigned long long)r->seq);continue;}
                unsigned wi=0;while(wi<GCP_W&&watches[wi].active)wi++;if(wi==GCP_W)break;
                MEMORY_BASIC_INFORMATION mi;bool valid=true;
                for(uint64_t a=lo;a<hi;a+=4096)if(!VirtualQuery((void *)(uintptr_t)a,&mi,sizeof mi)||mi.State!=MEM_COMMIT||mi.Protect!=PAGE_READWRITE){valid=false;break;}
                if(!valid){TRACE_LOG("[gcp-arm-skip] id %llu protection_conflict\n",(unsigned long long)r->id);continue;}
                Watch *w=&watches[wi];*w=(Watch){*r,lo,hi,plat_now_ns(),fence_id,true,false};DWORD old;
                if(!VirtualProtect((void *)(uintptr_t)lo,hi-lo,PAGE_NOACCESS,&old)){w->active=false;continue;}
                TRACE_LOG("[gcp-arm] %llu id %llu kind %u fence %llu seq %llu shader %llx addr %llx size %llu pages %llu applied %llu\n",(unsigned long long)w->arm,(unsigned long long)r->id,r->kind,(unsigned long long)fence_id,(unsigned long long)r->seq,(unsigned long long)r->shader,(unsigned long long)r->addr,(unsigned long long)r->size,(unsigned long long)((hi-lo)/4096),(unsigned long long)r->applied);
            }pick+=mode==3?1:4;
        }
        unsigned k=0;for(unsigned i=0;i<nr;i++)if(results[i].seq>seq)results[k++]=results[i];nr=k;
    }
    ReleaseSRWLockExclusive(&lock);
}
bool gcp_fault(uint64_t addr,uint64_t pc,bool write) {
    if(!mode)return false;
    AcquireSRWLockExclusive(&lock);
    for(unsigned i=0;i<GCP_W;i++){Watch *w=&watches[i];if(!w->active||addr<w->lo||addr>=w->hi)continue;
        open_watch(w);KThread *t=kernel_current_thread();MEMORY_BASIC_INFORMATION mi;uint64_t base=VirtualQuery((void *)(uintptr_t)pc,&mi,sizeof mi)?(uint64_t)(uintptr_t)mi.AllocationBase:0;
        TRACE_LOG("[gcp-hit] %llu id %llu kind %u fence %llu current_fence %llu addr %llx in_range %d write %d thread %x lr %llx pc %llx module %llx gpu_binding %d\n",(unsigned long long)plat_now_ns(),(unsigned long long)w->r.id,w->r.kind,(unsigned long long)w->fence,(unsigned long long)fence_id,(unsigned long long)addr,addr>=w->r.addr&&addr<w->r.addr+w->r.size,write,t?t->handle:0,(unsigned long long)(t?t->cpu.x[30]:0),(unsigned long long)pc,(unsigned long long)base,w->gpu);
        if(w->r.verify && !write && addr>=w->r.addr && addr<w->r.addr+w->r.size){
            uint64_t actual=row_hash((const void *)(uintptr_t)w->r.addr,w->r.row,w->r.rows,w->r.stride);
            TRACE_LOG("[gcp-readback-verify] ns %llu id %llu seq %llu producer_frame %llu shader %llx addr %llx payload %llu guest %u expected %llx actual %llx match %u\n",
                (unsigned long long)plat_now_ns(),(unsigned long long)w->r.id,(unsigned long long)w->r.seq,(unsigned long long)w->r.frame,
                (unsigned long long)w->r.shader,(unsigned long long)w->r.addr,(unsigned long long)w->r.payload,t!=NULL,
                (unsigned long long)w->r.expected,(unsigned long long)actual,actual==w->r.expected);
            if(actual!=w->r.expected)FATAL("A1b readback generation mismatch at first reader\n");
        }
        ReleaseSRWLockExclusive(&lock);return true;
    }
    ReleaseSRWLockExclusive(&lock);return false;
}
#else
bool gcp_active(void){return false;}
void gcp_result(unsigned a,uint64_t b,uint64_t c,uint64_t d,uint64_t e,uint64_t f,uint64_t g){}
void gcp_applied(unsigned a,uint64_t b,uint64_t c,uint64_t d){}
void gcp_readback_expected(uint64_t a,uint64_t b,const void*c,uint32_t d,uint32_t e,uint64_t f){}
void gcp_boundary(uint64_t a,uint64_t b){}
void gcp_gpu_use(const char*a,uint64_t b,uint64_t c,uint64_t d,uint64_t e){}
void gcp_fence(bool a,uint64_t b){}
void gcp_return(uint64_t a,uint64_t b,uint64_t c,uint64_t d,uint64_t e){}
void gcp_layout(uint64_t a,uint64_t b,uint64_t c,uint64_t d,uint64_t e){}
bool gcp_fault(uint64_t a,uint64_t b,bool c){return false;}
#endif
