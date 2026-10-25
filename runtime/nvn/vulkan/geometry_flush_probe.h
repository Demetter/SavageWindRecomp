#pragma once

// Diagnostic only: compare explicit NVN flush notifications with the existing
// OS dirty-page result. Do not change/reset that result or use this as proof that
// notifications are sufficient in other scenes. Concurrent writes after the
// snapshot can appear uncovered; a notification need not be the latest write.
#define GF_WORDS ((HEAP_MAX >> 12) / 64)
static _Atomic uint64_t gf_bits[GF_WORDS], gf_calls, gf_bytes;
static uint64_t gf_snapshot[GF_WORDS], gf_snapshot_calls, gf_snapshot_bytes;

static bool geometry_flush_probe_on(void) {
    static _Atomic int enabled = -1;
    int value = atomic_load_explicit(&enabled, memory_order_relaxed);
    if (value < 0) {
        const char *e = getenv("SAVAGEWIND_VK_GEOMETRY_FLUSH_PROBE");
        value = e && atoi(e) != 0;
        atomic_store_explicit(&enabled, value, memory_order_relaxed);
    }
    return value != 0;
}

void backend_pool_written(uint64_t addr, uint64_t size) {
    if (!geometry_flush_probe_on()) return;
    atomic_fetch_add_explicit(&gf_calls, 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&gf_bytes, size, memory_order_relaxed);
    if (!size || addr + size < addr) return;
    uint64_t end = addr + size;
    if (addr < HEAP_BASE) addr = HEAP_BASE;
    if (end > HEAP_BASE + HEAP_MAX) end = HEAP_BASE + HEAP_MAX;
    if (addr >= end) return;
    uint64_t first = (addr - HEAP_BASE) >> 12, last = (end - 1 - HEAP_BASE) >> 12;
    for (uint64_t p = first; p <= last;) {
        uint64_t w = p >> 6, stop = ((w << 6) + 63) < last ? ((w << 6) + 63) : last;
        uint64_t mask = (~0ull << (p & 63)) & (~0ull >> (63 - (stop & 63)));
        atomic_fetch_or_explicit(&gf_bits[w], mask, memory_order_relaxed);
        p = stop + 1;
    }
}

static void geometry_flush_probe_begin(void) {
    if (!geometry_flush_probe_on()) return;
    for (size_t i = 0; i < GF_WORDS; i++)
        gf_snapshot[i] = atomic_exchange_explicit(&gf_bits[i], 0, memory_order_relaxed);
    gf_snapshot_calls = atomic_exchange_explicit(&gf_calls, 0, memory_order_relaxed);
    gf_snapshot_bytes = atomic_exchange_explicit(&gf_bytes, 0, memory_order_relaxed);
}

static void geometry_flush_probe_end(const uint64_t *dirty, size_t n) {
    if (!geometry_flush_probe_on() || !vq_sparse || n == SIZE_MAX || n >= vram_pages_max) return;
    static uint64_t queries, pages, uncovered, calls, bytes, reported, missing[8];
    static unsigned nmissing;
    queries++;
    calls += gf_snapshot_calls;
    bytes += gf_snapshot_bytes;
    for (size_t i = 0; i < n; i++) {
        uint64_t p = (dirty[i] - HEAP_BASE) >> 12;
        if (p >= (HEAP_MAX >> 12)) continue;
        pages++;
        if (!(gf_snapshot[p >> 6] & (1ull << (p & 63)))) {
            uncovered++;
            bool seen = false;
            for (unsigned k = 0; k < nmissing; k++) seen |= missing[k] == dirty[i];
            if (!seen && nmissing < 8) missing[nmissing++] = dirty[i];
        }
    }
    if (vq_frame >= reported + 600) {
        LOG("[geometry-flush] frame %llu queries %llu dirty %llu uncovered %llu flush_calls %llu flush_bytes %llu missing",
            (unsigned long long)vq_frame, (unsigned long long)queries, (unsigned long long)pages,
            (unsigned long long)uncovered, (unsigned long long)calls, (unsigned long long)bytes);
        for (unsigned k = 0; k < nmissing; k++) LOG(" %llx", (unsigned long long)missing[k]);
        LOG("\n");
        reported = vq_frame;
        queries = pages = uncovered = calls = bytes = nmissing = 0;
    }
}
