// The bounded Windows 11 unwinder checks stack limits before following saved
// frames. The older RtlVirtualUnwind may touch a guard page before a caller can
// inspect its output. Profiling must never crash the process it observes.
#pragma once
#include <windows.h>
#include <stdbool.h>
#include <stdint.h>

typedef LONG (NTAPI *SampleUnwind2)(DWORD, DWORD64, DWORD64, PRUNTIME_FUNCTION,
    PCONTEXT, PBOOLEAN, PVOID *, PDWORD64, PKNONVOLATILE_CONTEXT_POINTERS,
    PDWORD64, PDWORD64, PEXCEPTION_ROUTINE *, DWORD);
static SampleUnwind2 sample_unwind2;
static bool sample_unwind_init(void) {
    sample_unwind2 = (SampleUnwind2)(void *)GetProcAddress(GetModuleHandleA("ntdll.dll"), "RtlVirtualUnwind2");
    return sample_unwind2 != NULL;
}
static bool sample_unwind_frame(CONTEXT *ctx, DWORD64 base, PRUNTIME_FUNCTION function, DWORD64 lo, DWORD64 hi) {
    if (!sample_unwind2 || hi < lo || hi - lo < 8 || ctx->Rsp < lo || ctx->Rsp > hi - 8) return false;
    void *data = NULL;
    DWORD64 frame = 0, previous = ctx->Rsp;
    LONG result = sample_unwind2(UNW_FLAG_NHANDLER, base, ctx->Rip, function, ctx,
                                NULL, &data, &frame, NULL, &lo, &hi, NULL, 0);
    return result == 0 && ctx->Rsp > previous && ctx->Rsp <= hi;
}
static bool sample_unwind_step(CONTEXT *ctx, DWORD64 lo, DWORD64 hi) {
    DWORD64 base = 0;
    PRUNTIME_FUNCTION function = RtlLookupFunctionEntry(ctx->Rip, &base, NULL);
    return sample_unwind_frame(ctx, base, function, lo, hi);
}
