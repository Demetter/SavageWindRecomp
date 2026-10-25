#pragma once
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "geometry_watch.h"

// One caller, persistent workers. All queries are non-resetting and cover
// disjoint page-aligned ranges. Reset/filter happens only after every query has
// completed, exactly as for the serial path. SIZE_MAX requires a full refresh.
#define WWP_THREADS 4
#define WWP_SPANS 64
typedef struct {
    GeometryWatchSpan spans[WWP_SPANS];
    unsigned nspans;
    uint64_t *pages;
    size_t capacity, result;
    double ms;
} WriteWatchJob;
typedef struct { double wall_ms, longest_ms; unsigned calls; uint64_t bytes; } WriteWatchStats;
static WriteWatchJob wwp_job[WWP_THREADS];
static pthread_mutex_t wwp_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t wwp_cv = PTHREAD_COND_INITIALIZER;
static unsigned wwp_generation, wwp_active, wwp_pending, wwp_initialized;
static bool wwp_ready[WWP_THREADS];

static void *wwp_worker(void *arg) {
    unsigned id=(unsigned)(uintptr_t)arg, seen=0;
    WriteWatchJob *j=&wwp_job[id];
#ifdef WWP_REGISTER_THREAD
    WWP_REGISTER_THREAD();
#endif
    pthread_mutex_lock(&wwp_lock);
    // A newly added worker must ignore requests completed before it existed.
    seen=wwp_generation;
    wwp_ready[id]=true;
    pthread_cond_broadcast(&wwp_cv);
    for (;;) {
        while (seen==wwp_generation) pthread_cond_wait(&wwp_cv,&wwp_lock);
        seen=wwp_generation;
        if (id>=wwp_active) continue;
        pthread_mutex_unlock(&wwp_lock);
        uint64_t start=plat_now_ns();
        size_t n=0;
        for (unsigned i=0;i<j->nspans;i++) {
            size_t k=plat_peek_written_pages(j->spans[i].lo,j->spans[i].hi-j->spans[i].lo,j->pages+n,j->capacity-n);
            if (k==SIZE_MAX || k>=j->capacity-n) { n=SIZE_MAX; break; }
            n+=k;
        }
        j->result=n; j->ms=(plat_now_ns()-start)/1e6;
        pthread_mutex_lock(&wwp_lock);
        wwp_pending--;
        pthread_cond_broadcast(&wwp_cv);
    }
    return NULL;
}

static bool wwp_partition(const GeometryWatchSpan *spans, unsigned nspans, unsigned workers, uint64_t *bytes) {
    if (!workers || workers>WWP_THREADS || nspans>WWP_SPANS) return false;
    uint64_t total=0;
    for (unsigned i=0;i<nspans;i++) {
        if (spans[i].hi<spans[i].lo || ((spans[i].lo|spans[i].hi)&4095) ||
            (i && spans[i].lo<spans[i-1].hi)) return false;
        total+=spans[i].hi-spans[i].lo;
    }
    uint64_t quota=((total/workers+4095)&~4095ull), used=0;
    if (!quota) quota=4096;
    unsigned w=0;
    for (unsigned i=0;i<workers;i++) wwp_job[i].nspans=0;
    for (unsigned i=0;i<nspans;i++) {
        uint64_t lo=spans[i].lo, hi=spans[i].hi;
        while (lo<hi) {
            if (used>=quota && w+1<workers) { w++; used=0; }
            uint64_t length=hi-lo;
            if (w+1<workers && length>quota-used) length=quota-used;
            WriteWatchJob *j=&wwp_job[w];
            if (j->nspans==WWP_SPANS) return false;
            j->spans[j->nspans++]=(GeometryWatchSpan){lo,lo+length};
            lo+=length; used+=length;
        }
    }
    *bytes=total;
    return true;
}

static size_t wwp_query(const GeometryWatchSpan *spans, unsigned nspans, unsigned workers,
                         uint64_t *out, size_t capacity, WriteWatchStats *stats) {
    uint64_t bytes=0;
    if (!capacity || !wwp_partition(spans,nspans,workers,&bytes)) return SIZE_MAX;
    // Start all helpers before publishing the first request. The ready handshake
    // prevents a late thread from mistaking that request for an old generation.
    while (wwp_initialized<workers) {
        // Thread creation is rare (first use, or a larger requested pool).
        pthread_t thread;
        unsigned id=wwp_initialized;
        wwp_job[id].result=SIZE_MAX;
        pthread_mutex_lock(&wwp_lock);
        if (pthread_create(&thread,NULL,wwp_worker,(void *)(uintptr_t)id)) {
            pthread_mutex_unlock(&wwp_lock); return SIZE_MAX;
        }
        pthread_detach(thread);
        while (!wwp_ready[id]) pthread_cond_wait(&wwp_cv,&wwp_lock);
        pthread_mutex_unlock(&wwp_lock);
        wwp_initialized++;
    }
    for (unsigned i=0;i<workers;i++) if (wwp_job[i].capacity<capacity) {
        uint64_t *p=realloc(wwp_job[i].pages,capacity*sizeof *p);
        if (!p) return SIZE_MAX;
        wwp_job[i].pages=p; wwp_job[i].capacity=capacity;
    }
    pthread_mutex_lock(&wwp_lock);
    uint64_t start=plat_now_ns();
    wwp_active=wwp_pending=workers; wwp_generation++;
    pthread_cond_broadcast(&wwp_cv);
    while (wwp_pending) pthread_cond_wait(&wwp_cv,&wwp_lock);
    double ms=(plat_now_ns()-start)/1e6;
    pthread_mutex_unlock(&wwp_lock);
    size_t n=0; double longest=0; unsigned calls=0;
    for (unsigned i=0;i<workers;i++) {
        WriteWatchJob *j=&wwp_job[i];
        if (j->result==SIZE_MAX || j->result>capacity-n) return SIZE_MAX;
        if (out && j->result) memcpy(out+n,j->pages,j->result*sizeof *out);
        n+=j->result; calls+=j->nspans;
        if (j->ms>longest) longest=j->ms;
    }
    if (stats) *stats=(WriteWatchStats){ms,longest,calls,bytes};
    return n;
}
