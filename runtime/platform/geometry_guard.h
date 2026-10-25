// Windows CPU-write tracking for the renderer's geometry mirror. Kept separate
// so the real protection/fault/I/O protocol can be exercised without the game.
// All protection transitions and dirty consumption share one lock. In
// particular a fault must not unprotect a page after a concurrent reset without
// leaving it dirty. Never allocate, log, or touch guest bytes inside this lock.
#pragma once
#include <windows.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static SRWLOCK gg_lock = SRWLOCK_INIT;
static volatile LONG64 gg_base_atomic;
static uint64_t gg_base, gg_size;
static size_t gg_pages, gg_words;
static uint64_t *gg_known, *gg_dirty, *gg_armed, *gg_blocked, *gg_owned;
static uint32_t *gg_io;
static volatile uint64_t gg_faults;  // diagnostic counter, also observed across VEH entry in the OS test

static bool gg_bit(const uint64_t *b, size_t p) { return (b[p >> 6] >> (p & 63)) & 1; }
static void gg_set(uint64_t *b, size_t p) { b[p >> 6] |= 1ull << (p & 63); }
static void gg_clear(uint64_t *b, size_t p) { b[p >> 6] &= ~(1ull << (p & 63)); }

static bool gg_open(size_t p) {
    if (!gg_bit(gg_armed, p)) return true;
    DWORD old;
    if (!VirtualProtect((void *)(uintptr_t)(gg_base + (p << 12)), 4096, PAGE_READWRITE, &old)) return false;
    gg_clear(gg_armed, p);
    return true;
}

// Called while the guest is stopped, before decommit or a heap-size change.
static void gg_reset(void) {
    AcquireSRWLockExclusive(&gg_lock);
    for (size_t w = 0; w < gg_words; w++)
        for (uint64_t b = gg_armed[w]; b; b &= b - 1)
            gg_open((w << 6) + __builtin_ctzll(b));
    InterlockedExchange64(&gg_base_atomic, 0);
    free(gg_known); free(gg_dirty); free(gg_armed); free(gg_blocked); free(gg_owned); free(gg_io);
    gg_known = gg_dirty = gg_armed = gg_blocked = gg_owned = NULL; gg_io = NULL;
    gg_base = gg_size = gg_pages = gg_words = gg_faults = 0;
    ReleaseSRWLockExclusive(&gg_lock);
}

static bool gg_init(uint64_t base, uint64_t size) {
    if (!base || !size || (base & 4095) || (size & 4095)) return false;
    if ((uint64_t)InterlockedCompareExchange64(&gg_base_atomic, 0, 0) == base && gg_size == size) return true;
    // Renderer calls this between lists; heap growth waits for previous work.
    gg_reset();
    size_t pages = (size_t)(size >> 12), words = (pages + 63) / 64;
    uint64_t *known = calloc(words, 8), *dirty = calloc(words, 8), *armed = calloc(words, 8), *blocked = calloc(words, 8);
    uint64_t *owned = calloc(words, 8);
    uint32_t *io = calloc(pages, sizeof *io);
    if (!known || !dirty || !armed || !blocked || !owned || !io) {
        free(known); free(dirty); free(armed); free(blocked); free(owned); free(io); return false;
    }
    AcquireSRWLockExclusive(&gg_lock);
    gg_base = base; gg_size = size; gg_pages = pages; gg_words = words;
    gg_known = known; gg_dirty = dirty; gg_armed = armed; gg_blocked = blocked; gg_owned = owned; gg_io = io;
    InterlockedExchange64(&gg_base_atomic, (LONG64)base);
    ReleaseSRWLockExclusive(&gg_lock);
    return true;
}

static bool gg_fault(uint64_t addr, bool write) {
    if (!write || !InterlockedCompareExchange64(&gg_base_atomic, 0, 0)) return false;
    AcquireSRWLockExclusive(&gg_lock);
    bool handled = false;
    if (addr >= gg_base && addr - gg_base < gg_size) {
        size_t p = (size_t)((addr - gg_base) >> 12);
        if (gg_bit(gg_known, p) && !gg_bit(gg_blocked, p)) {
            // Two CPUs may have faulted before either got the lock. The second
            // still marks dirty even if the first already opened the page.
            gg_set(gg_dirty, p);
            handled = gg_open(p);
            gg_faults += handled;
        }
    }
    ReleaseSRWLockExclusive(&gg_lock);
    return handled;
}

static bool gg_range(uint64_t addr, uint64_t size, size_t *lo, size_t *hi) {
    if (!gg_size || !size || addr > UINT64_MAX - size || addr + size <= gg_base || addr >= gg_base + gg_size) return false;
    *lo = addr <= gg_base ? 0 : (size_t)((addr - gg_base) >> 12);
    uint64_t end = addr + size;
    *hi = end >= gg_base + gg_size ? gg_pages : (size_t)((end - gg_base + 4095) >> 12);
    return true;
}

// Kernel I/O does not invoke the CPU fault handler when its destination is
// read-only. Pin the destination writable for the ENTIRE operation, including
// concurrent queries. End marks it dirty again even if a query ran during I/O.
static void gg_io_range(uint64_t addr, uint64_t size, bool begin) {
    if (!InterlockedCompareExchange64(&gg_base_atomic, 0, 0)) return;
    AcquireSRWLockExclusive(&gg_lock);
    size_t lo, hi;
    if (gg_range(addr, size, &lo, &hi)) for (size_t p = lo; p < hi; p++) {
        if (begin) { gg_io[p]++; gg_open(p); }
        else if (gg_io[p]) gg_io[p]--;
        if (gg_bit(gg_known, p)) gg_set(gg_dirty, p);
    }
    ReleaseSRWLockExclusive(&gg_lock);
}

// Explicit guest/debug protections take precedence. Such pages stay on the
// conservative refresh-every-query path until the tracker is reset.
static void gg_exclude(uint64_t addr, uint64_t size) {
    if (!InterlockedCompareExchange64(&gg_base_atomic, 0, 0)) return;
    AcquireSRWLockExclusive(&gg_lock);
    size_t lo, hi;
    if (gg_range(addr, size, &lo, &hi)) for (size_t p = lo; p < hi; p++) {
        gg_open(p); gg_set(gg_blocked, p); gg_set(gg_dirty, p);
    }
    ReleaseSRWLockExclusive(&gg_lock);
}

// Return an ascending list of dirty/new geometry pages. No hashes, sampling,
// delayed refresh, or OS dirty-bit resets. A caller must refresh every returned
// page after this call, just as for GetWriteWatch. Overflow returns SIZE_MAX.
static size_t gg_collect(const uint64_t *geometry, size_t pages, uint64_t *out, size_t cap) {
    AcquireSRWLockExclusive(&gg_lock);
    if (!gg_size || pages > gg_pages) { ReleaseSRWLockExclusive(&gg_lock); return SIZE_MAX; }
    size_t n = 0;
    size_t words = (pages + 63) / 64;
    for (size_t w = 0; w < words; w++) {
        uint64_t want = geometry[w];
        if (w == words - 1 && (pages & 63)) want &= (1ull << (pages & 63)) - 1;
        for (uint64_t b = gg_known[w] & ~want; b; b &= b - 1)
            gg_open((w << 6) + __builtin_ctzll(b));
        gg_owned[w] &= want;
        uint64_t changed = want & (~gg_known[w] | gg_dirty[w] | gg_blocked[w]);
        gg_known[w] = want;
        gg_dirty[w] &= ~want;
        for (uint64_t b = want; b; b &= b - 1) {
            size_t p = (w << 6) + __builtin_ctzll(b);
            uint64_t bit = 1ull << (p & 63);
            if (gg_io[p]) { changed |= bit; gg_set(gg_dirty, p); }
            else if (!gg_bit(gg_armed, p) && !gg_bit(gg_blocked, p)) {
                MEMORY_BASIC_INFORMATION mi; DWORD old;
                void *addr = (void *)(uintptr_t)(gg_base + (p << 12));
                // Never relax an application's original protection, and never
                // commit a page merely because a geometry range mentions it.
                // Once checked, this tracker's fault/I/O path is the only
                // owner changing RW/RO. Explicit protection changes exclude
                // the page, and decommit/reboot resets the entire tracker.
                // VirtualQuery scans the following same-protection region;
                // repeating it after every store caused periodic ~5 ms stalls.
                // New pages (including first seen during pinned I/O) still
                // require the original-protection check.
                if ((!gg_bit(gg_owned, p) &&
                     (!VirtualQuery(addr, &mi, sizeof mi) || mi.State != MEM_COMMIT || mi.Protect != PAGE_READWRITE)) ||
                    !VirtualProtect(addr, 4096, PAGE_READONLY, &old)) {
                    gg_set(gg_blocked, p); changed |= bit;
                } else { gg_set(gg_armed, p); gg_set(gg_owned, p); }
            }
        }
        for (uint64_t b = changed; b; b &= b - 1) {
            if (n < cap) out[n] = gg_base + (((uint64_t)w << 6) + __builtin_ctzll(b)) * 4096;
            n++;
        }
    }
    ReleaseSRWLockExclusive(&gg_lock);
    return n > cap ? SIZE_MAX : n;
}
