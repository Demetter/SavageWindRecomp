#pragma once
#include <stdint.h>
#include <stdbool.h>
// Diagnostic only. No wait elision, result deferral, or shader changes.
enum { GCP_READBACK=1, GCP_QUERY, GCP_GMEM_WRITE_BOUND, GCP_HEAP_READ,
       GCP_VERTEX_READ, GCP_UBO_READ, GCP_GMEM_UNKNOWN };
bool gcp_active(void);
void gcp_result(unsigned kind, uint64_t seq, uint64_t frame, uint64_t shader,
                uint64_t addr, uint64_t size, uint64_t payload);
void gcp_applied(unsigned kind, uint64_t seq, uint64_t addr, uint64_t hash);
void gcp_readback_expected(uint64_t seq,uint64_t addr,const void *gpu_rows,uint32_t row,uint32_t rows,uint64_t stride);
void gcp_boundary(uint64_t last, uint64_t done);
void gcp_gpu_use(const char *kind, uint64_t seq, uint64_t shader, uint64_t addr, uint64_t size);
void gcp_fence(bool end, uint64_t seq);
void gcp_return(uint64_t seq, uint64_t start, uint64_t queued, uint64_t gpu_start, uint64_t gpu_end);
void gcp_layout(uint64_t addr, uint64_t row, uint64_t rows, uint64_t stride, uint64_t shader);
bool gcp_fault(uint64_t addr, uint64_t pc, bool write);
