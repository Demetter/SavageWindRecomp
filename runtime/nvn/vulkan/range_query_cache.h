#pragma once
#include <stdbool.h>
#include <stdint.h>

// Producer-owned memo of range-overlap decisions, not guest-memory contents.
// Every mutation of the interval set and every reset must change generation.
typedef struct { uint64_t lo, hi, generation; bool overlaps; } RangeQueryEntry;
typedef struct { RangeQueryEntry entries[256]; } RangeQueryCache;

static unsigned range_query_slot(uint64_t lo, uint64_t hi) {
    return (unsigned)(((lo >> 4) ^ (hi * UINT64_C(0x9e3779b97f4a7c15))) >> 8) & 255;
}
static bool range_query_cached(const RangeQueryCache *cache, uint64_t generation,
                               uint64_t lo, uint64_t hi, bool *overlaps) {
    const RangeQueryEntry *e = &cache->entries[range_query_slot(lo, hi)];
    if (!generation || e->generation != generation || e->lo != lo || e->hi != hi) return false;
    *overlaps = e->overlaps;
    return true;
}
static void range_query_remember(RangeQueryCache *cache, uint64_t generation,
                                 uint64_t lo, uint64_t hi, bool overlaps) {
    cache->entries[range_query_slot(lo, hi)] = (RangeQueryEntry){lo, hi, generation, overlaps};
}
