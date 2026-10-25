// Windows write-watch semantics and the exact-page reset plan used by the renderer.
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <windows.h>
#include "../runtime/nvn/vulkan/geometry_watch.h"

static unsigned calls, fail_at;
static uint64_t reset_lo[8], reset_size[8];
static bool fake_reset(uint64_t lo, uint64_t size) {
    reset_lo[calls] = lo; reset_size[calls] = size;
    return ++calls != fail_at;
}
static bool real_reset(uint64_t lo, uint64_t size) {
    return ResetWriteWatch((void *)(uintptr_t)lo, size) == 0;
}
static size_t peek(void *p, uint64_t *pages, size_t cap) {
    ULONG_PTR n = cap; ULONG gran = 0;
    assert(!GetWriteWatch(0, p, 65536, (void **)pages, &n, &gran));
    assert(gran == 4096 && n < cap);
    return n;
}
static bool contains(const uint64_t *p, size_t n, uint64_t a) {
    for (size_t i = 0; i < n; i++) if (p[i] == a) return true;
    return false;
}
int main(void) {
    uint64_t layout_bits[6] = {1, 0, 8, 0, 0, 1};
    GeometryWatchSpan spans[6];
    uint64_t region_size = 5 * 262144ull + 4096;
    assert(geometry_watch_plan(layout_bits, 0x10000, region_size, 1, spans, 6) == 3);
    assert(spans[0].lo == 0x10000 && spans[0].hi == 0x50000);
    assert(spans[2].lo == 0x150000 && spans[2].hi == 0x151000);
    assert(geometry_watch_plan(layout_bits, 0x10000, region_size, 2, spans, 6) == 2);
    assert(spans[0].hi == 0xd0000);
    assert(geometry_watch_plan(layout_bits, 0x10000, region_size, 3, spans, 6) == 1);
    assert(spans[0].hi == 0x151000);
    spans[1].lo = 123;
    assert(geometry_watch_plan(layout_bits, 0x10000, region_size, 1, spans, 1) == 3);
    assert(spans[1].lo == 123);  // insufficient output capacity never overruns
    memset(layout_bits, 0, sizeof layout_bits);
    assert(!geometry_watch_plan(layout_bits, 0x10000, region_size, 1, spans, 6));
    assert(!geometry_watch_plan(NULL, 0x10000, 0, 1, spans, 6));
    uint64_t bits = (1ull << 2) | (1ull << 3) | (1ull << 5);
    uint64_t pages[] = {0x15000, 0x12000, 0x11000, 0x13000, 0x12000, 0x1000, 0x12001, 0x90000};
    size_t n = geometry_watch_collect(pages, 8, 0x10000, 16, &bits, fake_reset);
    assert(n == 3 && calls == 2);
    assert(reset_lo[0] == 0x12000 && reset_size[0] == 8192);
    assert(reset_lo[1] == 0x15000 && reset_size[1] == 4096);
    calls = 0; fail_at = 2;
    assert(geometry_watch_collect(pages, n, 0x10000, 16, &bits, fake_reset) == SIZE_MAX);
    calls = 0; fail_at = 0; bits = 0;
    assert(!geometry_watch_collect(pages, n, 0x10000, 16, &bits, fake_reset) && !calls);

    volatile unsigned char *mem = VirtualAlloc(NULL, 65536, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE);
    assert(mem && real_reset((uintptr_t)mem, 65536));
    mem[4096] = 1;  // non-geometry: must remain marked through every sparse query
    mem[8192] = 2;
    mem[12288] = 3;
    uint64_t dirty[32];
    n = peek((void *)mem, dirty, 32);
    assert(n == 3);
    mem[8192] = 42;  // write between peek and reset: the subsequent copy must include it
    bits = (1ull << 2) | (1ull << 3);
    n = geometry_watch_collect(dirty, n, (uintptr_t)mem, 16, &bits, real_reset);
    assert(n == 2 && mem[8192] == 42 && mem[12288] == 3);
    unsigned char copies[8192];
    memcpy(copies, (const void *)(mem + 8192), sizeof copies);
    assert(copies[0] == 42 && copies[4096] == 3);
    mem[8192] = 99;  // after reset/copy: must be observed by the next query
    n = peek((void *)mem, dirty, 32);
    assert(n == 2 && contains(dirty, n, (uintptr_t)mem + 4096) && contains(dirty, n, (uintptr_t)mem + 8192));
    assert(!contains(dirty, n, (uintptr_t)mem + 12288));
    assert(VirtualFree((void *)mem, 0, MEM_RELEASE));
    puts("geometry write watch: span coverage, exact resets, fallback, unrelated and intervening writes passed");
    return 0;
}
