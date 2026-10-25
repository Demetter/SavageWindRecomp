#include <assert.h>
#include <stdio.h>
#include <windows.h>
static unsigned protection_queries;
static SIZE_T audited_query(LPCVOID addr, PMEMORY_BASIC_INFORMATION info, SIZE_T size) {
    protection_queries++;
    return VirtualQuery(addr, info, size);
}
#define VirtualQuery audited_query
#include "../runtime/platform/geometry_guard.h"
#undef VirtualQuery

static LONG CALLBACK handler(EXCEPTION_POINTERS *ep) {
    EXCEPTION_RECORD *e = ep->ExceptionRecord;
    return e->ExceptionCode == EXCEPTION_ACCESS_VIOLATION && e->NumberParameters >= 2 &&
        gg_fault(e->ExceptionInformation[1], e->ExceptionInformation[0] == 1) ?
        EXCEPTION_CONTINUE_EXECUTION : EXCEPTION_CONTINUE_SEARCH;
}
static bool has(uint64_t *out, size_t n, uint64_t addr) {
    for (size_t i = 0; i < n; i++) if (out[i] == addr) return true;
    return false;
}
static volatile unsigned char *mem;
static HANDLE go[8], done[8];
static volatile LONG quit;
static DWORD WINAPI writer(void *arg) {
    size_t i = (size_t)arg;
    for (;;) {
        WaitForSingleObject(go[i], INFINITE);
        if (InterlockedCompareExchange(&quit, 0, 0)) return 0;
        // Two writers per page, at distinct byte addresses.
        mem[(i / 2) * 4096 + i]++;
        SetEvent(done[i]);
    }
}
int main(void) {
    void *veh = AddVectoredExceptionHandler(1, handler);
    mem = VirtualAlloc(NULL, 65536, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE);
    assert(veh && mem && gg_init((uintptr_t)mem, 65536));
    uint64_t bits = 15, out[16];
    assert(gg_collect(&bits, 16, out, 16) == 4);
    assert(protection_queries == 4);
    assert(!gg_collect(&bits, 16, out, 16));
    mem[4096] = 37;
    size_t n = gg_collect(&bits, 16, out, 16);
    assert(n == 1 && has(out, n, (uintptr_t)mem + 4096));
    // Repeated writes cause one dirty page, reads never cause a fault.
    uint64_t faults = gg_faults;
    assert(mem[4096] == 37 && gg_faults == faults);
    mem[4096] = 38; mem[4097] = 39;
    assert(gg_faults == faults + 1 && gg_collect(&bits, 16, out, 16) == 1);
    assert(protection_queries == 4); // repeated writes must not rescan OS regions
    // Non-geometry writes stay outside the tracker. Late-discovered pages are
    // refreshed unconditionally on their first collection.
    mem[4 * 4096] = 9;
    assert(!gg_collect(&bits, 16, out, 16));
    bits |= 16;
    assert(gg_collect(&bits, 16, out, 16) == 1);
    bits = 15;
    assert(!gg_collect(&bits, 16, out, 16));
    mem[4 * 4096] = 10;

    // Real kernel I/O into a protected page, with a query while the destination
    // is pinned writable and nested overlapping users.
    HANDLE file = CreateFileA("build/codex-perf/geometry-guard-io.tmp", GENERIC_READ | GENERIC_WRITE,
        0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_DELETE_ON_CLOSE, NULL);
    assert(file != INVALID_HANDLE_VALUE);
    unsigned char bytes[4096]; memset(bytes, 0xa7, sizeof bytes); DWORD count;
    assert(WriteFile(file, bytes, sizeof bytes, &count, NULL) && count == sizeof bytes);
    SetFilePointer(file, 0, NULL, FILE_BEGIN);
    gg_io_range((uintptr_t)mem, 4096, true);
    gg_io_range((uintptr_t)mem + 16, 16, true);
    assert(gg_collect(&bits, 16, out, 16) == 1);
    assert(ReadFile(file, (void *)mem, 4096, &count, NULL) && count == 4096);
    gg_io_range((uintptr_t)mem + 16, 16, false);
    assert(gg_collect(&bits, 16, out, 16) == 1 && !gg_bit(gg_armed, 0));
    gg_io_range((uintptr_t)mem, 4096, false);
    assert(gg_collect(&bits, 16, out, 16) == 1 && gg_bit(gg_armed, 0));
    assert(!memcmp((const void *)mem, bytes, sizeof bytes));
    CloseHandle(file);

    // Library/vector writes that straddle page boundaries and recommit of an
    // already committed destination must leave both pages refreshable.
    memset((void *)(mem + 4096 - 32), 0x5a, 64);
    assert(gg_collect(&bits, 16, out, 16) == 2);
    gg_io_range((uintptr_t)mem + 8192, 4096, true);
    assert(VirtualAlloc((void *)(mem + 8192), 4096, MEM_COMMIT, PAGE_READWRITE));
    gg_io_range((uintptr_t)mem + 8192, 4096, false);
    assert(gg_collect(&bits, 16, out, 16) == 1);
    // A page first registered while a kernel writer already holds it stays
    // writable and is reported again after that writer completes.
    gg_io_range((uintptr_t)mem + 4 * 4096, 4096, true);
    bits |= 16;
    assert(gg_collect(&bits, 16, out, 16) == 1 && !gg_bit(gg_armed, 4));
    mem[4 * 4096] = 0x77;
    gg_io_range((uintptr_t)mem + 4 * 4096, 4096, false);
    assert(gg_collect(&bits, 16, out, 16) == 1 && gg_bit(gg_armed, 4));
    // Once removed, a newly discovered page must have its protection checked
    // again, even if the tracker owned it earlier.
    unsigned queries = protection_queries;
    bits = 15;
    assert(!gg_collect(&bits, 16, out, 16));
    DWORD rediscovery_old;
    assert(VirtualProtect((void *)(mem + 4 * 4096), 4096, PAGE_READONLY, &rediscovery_old));
    bits |= 16;
    assert(gg_collect(&bits, 16, out, 16) == 1 && !gg_bit(gg_armed, 4));
    assert(protection_queries == queries + 1 && gg_bit(gg_blocked, 4));
    assert(VirtualProtect((void *)(mem + 4 * 4096), 4096, PAGE_READWRITE, &rediscovery_old));
    bits = 15;

    // Multiwriter faults racing collection: every completed round must leave
    // each touched page reported, unless the racing query already consumed it.
    HANDLE threads[8];
    for (size_t i = 0; i < 8; i++) {
        go[i] = CreateEvent(NULL, FALSE, FALSE, NULL); done[i] = CreateEvent(NULL, FALSE, FALSE, NULL);
        threads[i] = CreateThread(NULL, 0, writer, (void *)i, 0, NULL);
        assert(go[i] && done[i] && threads[i]);
    }
    for (unsigned round = 0; round < 2000; round++) {
        gg_collect(&bits, 16, out, 16);
        for (int i = 0; i < 8; i++) SetEvent(go[i]);
        uint64_t seen = 0;
        n = gg_collect(&bits, 16, out, 16);
        for (size_t i = 0; i < n; i++) seen |= 1ull << ((out[i] - (uintptr_t)mem) >> 12);
        assert(WaitForMultipleObjects(8, done, TRUE, 10000) == WAIT_OBJECT_0);
        n = gg_collect(&bits, 16, out, 16);
        for (size_t i = 0; i < n; i++) seen |= 1ull << ((out[i] - (uintptr_t)mem) >> 12);
        assert(seen == 15);
    }
    InterlockedExchange(&quit, 1);
    for (int i = 0; i < 8; i++) SetEvent(go[i]);
    assert(WaitForMultipleObjects(8, threads, TRUE, 10000) == WAIT_OBJECT_0);
    for (int i = 0; i < 8; i++) CloseHandle(threads[i]), CloseHandle(go[i]), CloseHandle(done[i]);

    // Explicit protections are respected; an excluded page is copied each time.
    gg_exclude((uintptr_t)mem, 4096);
    DWORD old; assert(VirtualProtect((void *)mem, 4096, PAGE_READONLY, &old));
    assert(!gg_fault((uintptr_t)mem, true));
    assert(gg_collect(&bits, 16, out, 16) == 1);
    assert(VirtualProtect((void *)mem, 4096, PAGE_READWRITE, &old));
    gg_reset();
    assert(VirtualFree((void *)mem, 65536, MEM_DECOMMIT));
    assert(VirtualAlloc((void *)mem, 65536, MEM_COMMIT, PAGE_READWRITE));
    assert(gg_init((uintptr_t)mem, 65536));
    assert(gg_collect(&bits, 16, out, 1) == SIZE_MAX); // caller uses full refresh
    mem[8192] = 99;
    assert(gg_collect(&bits, 16, out, 16) == 1);
    gg_reset();
    assert(VirtualFree((void *)mem, 0, MEM_RELEASE));
    RemoveVectoredExceptionHandler(veh);
    puts("geometry guard: real faults, kernel I/O, nested I/O, 2000 multiwriter races, overflow and reboot passed");
    return 0;
}
