#pragma once
#include <stdint.h>
enum {A1D_PRODUCER,A1D_DESCRIPTOR,A1D_RECORDER,A1D_READY,A1D_DRAIN,A1D_FENCE,A1D_FENCE_QUEUE,A1D_FENCE_GPU,A1D_FENCE_CORE,A1D_SETS,A1D_HITS,A1D_COUNT};
extern int a1d_enabled;
extern __thread uint64_t a1d_local[A1D_COUNT];
void a1d_init(void);
void a1d_publish(void);
void a1d_add(unsigned metric,uint64_t value);
void a1d_frame(uint64_t frame);
