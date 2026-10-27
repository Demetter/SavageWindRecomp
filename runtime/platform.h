// Host OS layer: everything the runtime needs that differs between macOS and Windows.
// Implementations: runtime/platform/macos.c, runtime/platform/windows.c (the Makefile picks one).
#pragma once
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef _WIN32
#include "platform/win_compat.h"
#elif defined(__linux__)
#include "platform/linux_compat.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

// ---- guest address space
// Claim [lo, hi) for the guest at exactly that address, readable/writable (pages may be backed lazily).
bool plat_reserve_guest(uint64_t lo, uint64_t hi);
// Make sure [addr, addr+size) inside the guest window is backed (before the OS itself writes there, e.g.
// file reads); a no-op where the whole window is backed from the start.
void plat_commit(uint64_t addr, uint64_t size);
// In-place reboot: every page of [lo, hi) back to zero (and unbacked), the window still claimed.
bool plat_reset_guest(uint64_t lo, uint64_t hi);
// The 4K pages of [addr, addr+size) inside the guest window written since the last call (the first call: since the
// reservation), into pages[] (at most max); the count (max when there may have been more: then take the whole range
// as written), or SIZE_MAX where the OS can't tell. Only the CPU's writes count, not a GPU's.
size_t plat_written_pages(uint64_t addr, uint64_t size, uint64_t *pages, size_t max);
// Non-resetting query and an explicit reset of exactly the region that will be
// refreshed. A caller must copy every reset page after the reset; it must not
// clear writes to pages whose cached contents it retains.
size_t plat_peek_written_pages(uint64_t addr, uint64_t size, uint64_t *pages, size_t max);
bool plat_reset_written_pages(uint64_t addr, uint64_t size);
#ifdef _WIN32
// Optional precise geometry CPU-write tracker. Initialized before guest I/O;
// guarded file destinations stay writable until the kernel operation finishes.
bool plat_geometry_guard_active(void);
size_t plat_geometry_guard_collect(const uint64_t *bits, size_t pages, uint64_t *out, size_t cap);
bool plat_geometry_guard_pending(uint64_t addr);  // diagnostic concurrent-write audit
#endif
enum { PLAT_PROT_NONE = 0, PLAT_PROT_READ = 1, PLAT_PROT_RW = 3 };
int plat_protect(uint64_t addr, uint64_t size, int prot);  // 0 on success
#define PLAT_PAGE 0x4000ull  // largest page size we deal with (macOS 16K; multiple of Windows' 4K)

// ---- time
uint64_t plat_now_ns(void);                // monotonic
#ifdef _WIN32
uint64_t plat_thread_cycles(void);         // scheduled CPU cycles; not elapsed time
uint32_t plat_thread_id(void);
#else
static inline uint64_t plat_thread_cycles(void) { return 0; }
static inline uint32_t plat_thread_id(void) { return 0; }
#endif
void plat_sleep_until_ns(uint64_t t);      // precise wait for plat_now_ns() >= t
void plat_thread_realtime(uint64_t period_ns);  // periodic timing thread (vsync): don't coalesce its timers

// ---- address waits (the OS's futex: WaitOnAddress, futex, __ulock)
// Blocks while *addr == expected, until a wake on addr or timeout_ns passes (it may also return early); false on
// timeout.
bool plat_wait_u32(const uint32_t *addr, uint32_t expected, uint64_t timeout_ns);
void plat_wake_u32_all(const uint32_t *addr);

// ---- process
void plat_init(void);                                // first thing in main()
bool plat_exe_path(char *buf, size_t size);
void plat_exec(char *const argv[]);  // replace the process (NULL = same command line); returns only on failure
#if defined(_WIN32) || defined(__linux__)
void plat_terminate(void);  // end the process now, skipping DLL teardown (exit status 0)
#endif

// Child processes (game code generator and compiler, game-file setup). stdout + stderr go to log_path (NULL = /dev/null).
typedef intptr_t PlatProc;  // 0 = failed to start
PlatProc plat_spawn(const char *const argv[], const char *log_path, bool append_log);
int plat_proc_wait(PlatProc p, bool block);  // exit code, or -1 while still running (block = false)
void plat_proc_kill(PlatProc p);             // end it and what it started (then it's gone: no wait)
bool plat_remove_tree(const char *dir);      // rm -rf
#ifdef _WIN32
#define PLAT_EXE_SUFFIX ".exe"
#define PLAT_VENV_PYTHON "venv/Scripts/python.exe"  // relative to the venv's parent
#else
#define PLAT_EXE_SUFFIX ""
#define PLAT_VENV_PYTHON "venv/bin/python"
#endif

// ---- diagnostics
// Name of the host function containing pc (guest functions are f_<addr>), NULL if unknown.
const char *plat_symbol(uint64_t pc, uint64_t *offset);
void plat_print_backtrace(void);
void plat_fatal_notice(void);  // a fatal error is about to end the process: tell the user (Windows release: message box)
void plat_crash_handler_install(void);  // per thread where the OS needs it (alternate signal stack)
// Called by the crash handler (runtime/dispatch.c): true = fault handled (watchpoint), resume.
bool rt_fault_watch(uint64_t fault_addr, uint64_t host_pc);
extern int g_fault_is_write;  // set by the platform handler before rt_fault_watch: 1 write, 0 read (or unknown)
void rt_fault_report(const char *what, uint64_t fault_addr, uint64_t host_pc, uint64_t host_lr) __attribute__((noreturn));

#ifdef __cplusplus
}
#endif
