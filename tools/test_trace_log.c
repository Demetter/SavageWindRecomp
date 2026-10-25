// Real Windows producer/consumer queue: forced backpressure, wrap, ordering,
// lossless flush, output failure, and no file writes on a producer thread.
#include <windows.h>
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static DWORD writer_id;
static LONG writes, fail_write;
static size_t checked_write(const void *p, size_t size, size_t count, FILE *file) {
    DWORD id = GetCurrentThreadId();
    if (!writer_id) writer_id = id;
    assert(writer_id == id);
    InterlockedIncrement(&writes);
    Sleep(5); // Deliberately slower than the producers; capacity must not drop data.
    if (InterlockedCompareExchange(&fail_write, 0, 0)) return 0;
    return fwrite(p, size, count, file);
}
#define TRACE_LOG_CAPACITY (128u << 10)
#define fwrite checked_write
#include "../runtime/platform/trace_log.h"
#undef fwrite

enum { PRODUCERS = 4, RECORDS = 10000 };
static DWORD WINAPI produce(LPVOID p) {
    unsigned id = (unsigned)(uintptr_t)p;
    for (unsigned i = 0; i < RECORDS; i++) {
        char text[256];
        int n = snprintf(text, sizeof text, "%u %u %08x abcdefghijklmnopqrstuvwxyz0123456789\n", id, i, i * 2654435761u + id);
        trace_log_append(text, (size_t)n);
        if (!(i % 1000)) assert(trace_log_flush());
    }
    return 0;
}
int main(int argc, char **argv) {
    assert(argc == 2);
    FILE *file = fopen(argv[1], "w+b");
    assert(file && trace_log_init(file));
    trace_log_append("before\n", 7);
    assert(trace_log_flush());
    HANDLE threads[PRODUCERS];
    for (unsigned i = 0; i < PRODUCERS; i++) {
        threads[i] = CreateThread(NULL, 0, produce, (void *)(uintptr_t)i, 0, NULL);
        assert(threads[i]);
    }
    for (unsigned i = 0; i < 100; i++) {
        fprintf(file, "direct\n"); // Ordinary logging interleaved with queued chunks.
        Sleep(1);
    }
    assert(WaitForMultipleObjects(PRODUCERS, threads, TRUE, 30000) == WAIT_OBJECT_0);
    for (unsigned i = 0; i < PRODUCERS; i++) CloseHandle(threads[i]);
    assert(trace_log_flush());
    assert(trace_log.written == trace_log.put && trace_log.backpressure > 0 && writer_id != GetCurrentThreadId());
    InterlockedExchange(&fail_write, 1);
    trace_log_append("write failure\n", 14);
    assert(!trace_log_flush());
    rewind(file);
    char text[256], expected[256];
    assert(fgets(text, sizeof text, file) && !strcmp(text, "before\n"));
    unsigned counts[PRODUCERS] = {0}, total = 0, direct = 0;
    while (fgets(text, sizeof text, file)) {
        if (!strcmp(text, "direct\n")) { direct++; continue; }
        unsigned id, i;
        assert(sscanf(text, "%u %u", &id, &i) == 2 && id < PRODUCERS && i == counts[id]++);
        snprintf(expected, sizeof expected, "%u %u %08x abcdefghijklmnopqrstuvwxyz0123456789\n", id, i, i * 2654435761u + id);
        assert(!strcmp(text, expected));
        total++;
    }
    assert(total == PRODUCERS * RECORDS && direct == 100);
    for (unsigned i = 0; i < PRODUCERS; i++) assert(counts[i] == RECORDS);
    printf("PASS: %u ordered records; %ld writes on writer only; %llu backpressure waits; flush and I/O error covered\n",
           total, writes, (unsigned long long)trace_log.backpressure);
    fclose(file);
    return 0;
}
