#pragma once
#include <stdbool.h>
#include <stdint.h>
#include <limits.h>

// Whether one indexed draw can use another draw's vertex binding offsets by
// changing baseVertex. Instanced/constant streams cannot move in this scheme.
// A shader reading VertexIndex also needs the returned delta subtracted from
// its observed builtin. This helper does not approve reordering or resource reuse.
// Require nonnegative baseVertex and delta: moving a vertex binding forward and
// compensating with a negative index changes robust-buffer bounds. Actual RTX5090
// graphics tests returned zero vertices for that mathematically equal address.
// A caller may instead choose lower shared binding offsets before assembling a
// batch, then rebase every draw against those offsets with this same check.
static inline bool native_draw_vertex_rebase(uint32_t streams,
        const uint64_t baseline[17], const uint64_t current[17],
        const uint32_t stride[17], const uint32_t divisor[17],
        int32_t base_vertex, int32_t *rebased, int64_t *vertex_id_delta) {
    if (base_vertex < 0) return false; // accepting this needs a proven minimum index
    bool have_delta = false;
    int64_t delta = 0;
    for (unsigned i = 0; i < 17; i++) if (streams & (1u << i)) {
        if (divisor[i] || !stride[i]) {
            if (baseline[i] != current[i]) return false;
            continue;
        }
        bool negative = current[i] < baseline[i];
        uint64_t magnitude = negative ? baseline[i] - current[i] : current[i] - baseline[i];
        if (magnitude % stride[i]) return false;
        magnitude /= stride[i];
        if (magnitude > INT64_MAX) return false;
        int64_t d = negative ? -(int64_t)magnitude : (int64_t)magnitude;
        if (have_delta && d != delta) return false;
        delta = d; have_delta = true;
    }
    if (delta < 0 || delta > (int64_t)INT32_MAX - base_vertex) return false;
    *rebased = (int32_t)(base_vertex + delta);
    *vertex_id_delta = delta;
    return true;
}

// Both index ranges use a common buffer; choose the smaller binding offset and
// put the differences in firstIndex. The caller must separately verify type,
// buffer identity, draw range and indirect-command support.
static inline bool native_draw_index_rebase(uint64_t a, uint64_t b, unsigned bytes,
                                            uint64_t *binding, uint32_t *first_a, uint32_t *first_b) {
    if ((bytes != 2 && bytes != 4) || a % bytes || b % bytes) return false;
    uint64_t base = a < b ? a : b, da = (a - base) / bytes, db = (b - base) / bytes;
    if (da > UINT32_MAX || db > UINT32_MAX) return false;
    *binding = base; *first_a = (uint32_t)da; *first_b = (uint32_t)db;
    return true;
}
