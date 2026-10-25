// Windows high-volume telemetry. The producer never owns stderr's FILE lock or
// performs file I/O after startup. A full queue applies backpressure (no lost
// records); the counter makes that observer effect visible in the final log.
#pragma once
#include <windows.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef TRACE_LOG_CAPACITY
#define TRACE_LOG_CAPACITY (8u << 20)
#endif
#define TRACE_LOG_CHUNK (64u << 10)
static struct {
    SRWLOCK lock;
    CONDITION_VARIABLE work, progress;
    char *data;
    uint64_t put, taken, written, flush_target, backpressure;
    HANDLE thread;
    FILE *file;
    bool io_failed;
} trace_log;

static DWORD WINAPI trace_log_writer(LPVOID arg) {
    (void)arg;
    char chunk[TRACE_LOG_CHUNK];
    for (;;) {
        AcquireSRWLockExclusive(&trace_log.lock);
        while (trace_log.put == trace_log.taken)
            SleepConditionVariableSRW(&trace_log.work, &trace_log.lock, INFINITE, 0);
        // Coalesce telemetry instead of waking the filesystem for every present.
        // An explicit flush or a large batch wakes this wait immediately.
        if (trace_log.put - trace_log.taken < 16384 && trace_log.flush_target <= trace_log.written)
            SleepConditionVariableSRW(&trace_log.work, &trace_log.lock, 16, 0);
        size_t n = (size_t)(trace_log.put - trace_log.taken);
        if (n > sizeof chunk) n = sizeof chunk;
        size_t off = (size_t)(trace_log.taken % TRACE_LOG_CAPACITY), first = TRACE_LOG_CAPACITY - off;
        if (first > n) first = n;
        memcpy(chunk, trace_log.data + off, first);
        memcpy(chunk + first, trace_log.data, n - first);
        // Keep complete telemetry lines together when ordinary LOG calls share
        // stderr. Splitting a line across writes would let another log splice
        // bytes into its JSON or frame counter. Public trace records end in LF.
        if (n == sizeof chunk) {
            size_t end = n;
            while (end && chunk[end - 1] != '\n') end--;
            if (end) n = end;
        }
        trace_log.taken += n;
        WakeAllConditionVariable(&trace_log.progress);
        ReleaseSRWLockExclusive(&trace_log.lock);
        // I/O and its potentially slow CRT lock are outside the producer lock.
        bool failed = fwrite(chunk, 1, n, trace_log.file) != n;
        failed |= fflush(trace_log.file) != 0;
        AcquireSRWLockExclusive(&trace_log.lock);
        trace_log.io_failed |= failed;
        trace_log.written += n;
        WakeAllConditionVariable(&trace_log.progress);
        ReleaseSRWLockExclusive(&trace_log.lock);
    }
}

static bool trace_log_init(FILE *file) {
    trace_log.data = malloc(TRACE_LOG_CAPACITY);
    if (!trace_log.data) return false;
    // Commit every page before gameplay; malloc alone leaves first-touch faults.
    volatile char *pages = trace_log.data;
    for (size_t i = 0; i < TRACE_LOG_CAPACITY; i += 4096) pages[i] = 0;
    trace_log.file = file;
    trace_log.thread = CreateThread(NULL, 0, trace_log_writer, NULL, 0, NULL);
    if (!trace_log.thread) { free(trace_log.data); trace_log.data = NULL; return false; }
    SetThreadPriority(trace_log.thread, THREAD_PRIORITY_BELOW_NORMAL);
    return true;
}

static void trace_log_append(const char *text, size_t size) {
    AcquireSRWLockExclusive(&trace_log.lock);
    // A formatted record is atomic with respect to other producers. Telemetry
    // records are below 2 KB; the formatter handles larger records separately.
    while (trace_log.put - trace_log.taken + size > TRACE_LOG_CAPACITY) {
        trace_log.backpressure++;
        WakeConditionVariable(&trace_log.work);
        SleepConditionVariableSRW(&trace_log.progress, &trace_log.lock, INFINITE, 0);
    }
    bool was_empty = trace_log.put == trace_log.taken;
    size_t off = (size_t)(trace_log.put % TRACE_LOG_CAPACITY), first = TRACE_LOG_CAPACITY - off;
    if (first > size) first = size;
    memcpy(trace_log.data + off, text, first);
    memcpy(trace_log.data, text + first, size - first);
    trace_log.put += size;
    if (was_empty || trace_log.put - trace_log.taken >= 16384) WakeConditionVariable(&trace_log.work);
    ReleaseSRWLockExclusive(&trace_log.lock);
}

static bool trace_log_flush(void) {
    if (!trace_log.thread) return true;
    AcquireSRWLockExclusive(&trace_log.lock);
    uint64_t target = trace_log.put;
    if (target > trace_log.flush_target) trace_log.flush_target = target;
    WakeConditionVariable(&trace_log.work);
    while (trace_log.written < target) {
        if (!SleepConditionVariableSRW(&trace_log.progress, &trace_log.lock, 5000, 0)) {
            ReleaseSRWLockExclusive(&trace_log.lock);
            return false;
        }
    }
    bool ok = !trace_log.io_failed;
    ReleaseSRWLockExclusive(&trace_log.lock);
    return ok;
}
