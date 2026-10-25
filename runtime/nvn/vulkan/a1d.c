// Default-off measurement only; no rendering/scheduling decisions.
#include "a1d.h"
#include "../../rt.h"
#include "../../platform.h"
#include <stdatomic.h>
#include <stdlib.h>
#include <pthread.h>
int a1d_enabled;
__thread uint64_t a1d_local[A1D_COUNT];
static struct {_Alignas(64) _Atomic uint64_t value;} totals[A1D_COUNT];
static pthread_once_t once=PTHREAD_ONCE_INIT;
static void init(void){const char *e=getenv("SAVAGEWIND_A1D");a1d_enabled=e&&atoi(e);}
void a1d_init(void){pthread_once(&once,init);}
void a1d_add(unsigned m,uint64_t v){if(a1d_enabled)atomic_fetch_add_explicit(&totals[m].value,v,memory_order_relaxed);}
void a1d_publish(void){if(a1d_enabled)for(unsigned i=0;i<A1D_COUNT;i++)if(a1d_local[i]){a1d_add(i,a1d_local[i]);a1d_local[i]=0;}}
void a1d_frame(uint64_t frame){
 if(!a1d_enabled)return;a1d_publish();static uint64_t old[A1D_COUNT];uint64_t d[A1D_COUNT];
 for(unsigned i=0;i<A1D_COUNT;i++){uint64_t n=atomic_load_explicit(&totals[i].value,memory_order_relaxed);d[i]=n-old[i];old[i]=n;}
 TRACE_LOG("[a1d-frame] frame %llu ns %llu producer_ns %llu descriptor_ns %llu recorder_ns %llu ready_ns %llu drain_ns %llu fence_ns %llu fence_queue_ns %llu fence_gpu_ns %llu fence_core_ns %llu sets %llu hits %llu\n",
 (unsigned long long)frame,(unsigned long long)plat_now_ns(),(unsigned long long)d[0],(unsigned long long)d[1],(unsigned long long)d[2],(unsigned long long)d[3],(unsigned long long)d[4],(unsigned long long)d[5],(unsigned long long)d[6],(unsigned long long)d[7],(unsigned long long)d[8],(unsigned long long)d[9],(unsigned long long)d[10]);
}
