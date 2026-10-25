#include <assert.h>
#include <stdio.h>
#include <string.h>
#include "../runtime/nvn/vulkan/range_query_cache.h"

// Independent byte-coverage oracle, including interval-set mutations, slot
// collisions, zero-sized queries, reused guest addresses and reset generations.
int main(void) {
    RangeQueryCache cache = {0};
    unsigned char covered[4096] = {0};
    uint64_t generation = 1, rng = 0x123456789;
    unsigned hits = 0;
    for (unsigned i = 0; i < 200000; i++) {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        if (!(i % 31)) {
            unsigned lo = (unsigned)rng % sizeof covered;
            unsigned hi = lo + (unsigned)(rng >> 32) % (sizeof covered - lo);
            memset(covered + lo, 1, hi - lo); generation++;
        }
        if (!(i % 251)) { memset(covered, 0, sizeof covered); generation++; }
        uint64_t lo = (rng >> 16) % sizeof covered;
        uint64_t hi = lo + (rng >> 40) % (sizeof covered - lo);
        bool ref = false, got;
        for (uint64_t j = lo; j < hi; j++) ref |= covered[j] != 0;
        if (range_query_cached(&cache, generation, lo, hi, &got)) { assert(got == ref); hits++; }
        range_query_remember(&cache, generation, lo, hi, ref);
        assert(range_query_cached(&cache, generation, lo, hi, &got) && got == ref);
        assert(!range_query_cached(&cache, generation + 1, lo, hi, &got));
        assert(!range_query_cached(&cache, 0, lo, hi, &got));
    }
    // Force a slot collision and require both endpoints, not just the hash.
    uint64_t lo = UINT64_MAX - 4096, hi = UINT64_MAX;
    range_query_remember(&cache, generation, lo, hi, true);
    bool got;
    assert(range_query_cached(&cache, generation, lo, hi, &got) && got);
    uint64_t other = 0;
    while (range_query_slot(other, other + 1) != range_query_slot(lo, hi)) other++;
    range_query_remember(&cache, generation, other, other + 1, false);
    assert(!range_query_cached(&cache, generation, lo, hi, &got));
    puts("range queries: coverage oracle, generation invalidation, collisions and full-width endpoints passed");
    (void)hits;
    return 0;
}
