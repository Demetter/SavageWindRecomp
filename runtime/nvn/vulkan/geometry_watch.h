#pragma once
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct { uint64_t lo, hi; } GeometryWatchSpan;

// Cover every known geometry page, merging gaps in units of 64 pages. The last
// word may be partial. Return the required span count even when output is full.
static size_t geometry_watch_plan(const uint64_t *bits, uint64_t base, uint64_t size,
                                  size_t gap_words, GeometryWatchSpan *out, size_t cap) {
    size_t words = (size_t)((size + 262143) >> 18), lo = 0, hi = 0, n = 0;
    bool open = false;
    for (size_t w = 0; w <= words; w++) {
        if (w < words && !bits[w]) continue;
        if (open && (w == words || w - hi > gap_words)) {
            if (n < cap) {
                uint64_t end = (uint64_t)(hi + 1) << 18;
                out[n] = (GeometryWatchSpan){base + ((uint64_t)lo << 18), base + (end < size ? end : size)};
            }
            n++; open = false;
        }
        if (w == words) break;
        if (!open) lo = w, open = true;
        hi = w;
    }
    return n;
}

static int geometry_watch_compare(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return (x > y) - (x < y);
}

// Retain only dirty geometry, and reset exactly those pages. All returned pages
// MUST be copied after this call. A write between peek and reset is consequently
// included by that copy; a write after reset stays marked for the next refresh.
// Do not merge through gaps: clearing an unreturned page would lose its write.
// On any reset failure the caller must refresh all geometry, including pages
// reset by earlier successful calls. Input order and duplicates are not trusted.
static size_t geometry_watch_collect(uint64_t *pages, size_t n, uint64_t base, size_t npages,
                                     const uint64_t *geometry, bool (*reset)(uint64_t, uint64_t)) {
    size_t out = 0;
    for (size_t i = 0; i < n; i++) {
        uint64_t a = pages[i];
        if (a < base || (a & 4095)) continue;
        uint64_t p = (a - base) >> 12;
        if (p < npages && (geometry[p >> 6] & (1ull << (p & 63)))) pages[out++] = a;
    }
    qsort(pages, out, sizeof *pages, geometry_watch_compare);
    size_t unique = 0;
    for (size_t i = 0; i < out; i++)
        if (!unique || pages[i] != pages[unique - 1]) pages[unique++] = pages[i];
    for (size_t i = 0; i < unique;) {
        size_t first = i++;
        while (i < unique && pages[i] == pages[i - 1] + 4096) i++;
        if (!reset(pages[first], (uint64_t)(i - first) << 12)) return SIZE_MAX;
    }
    return unique;
}
