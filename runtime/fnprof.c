// Function-entry profiler. SAVAGEWIND_FNPROF=<from>:<to> counts guest function entries (every generated
// function calls FN_ENTER) between those presented frames, then writes build/fnprof.txt (addr count, most
// called first). Diffing two runs shows which game functions a change brings in; counts also show hot code.
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hle.h"
#include "platform.h"

volatile int g_fnprof;

#define NSLOT (1 << 18)
static struct { _Atomic uint64_t addr; _Atomic uint64_t count; _Atomic uint8_t who; } slots[NSLOT];

// SAVAGEWIND_FNPROF_WHO=<main handle hex>:<job worker handles hex, comma-separated>: each function is also marked with
// the kinds of thread that entered it (1 the main thread, 2 a job worker, 4 any other), written as a third column
// (which game code runs in BOTW's jobs)
static uint32_t who_main, who_jobs[16];
static int who_n = -1;
extern uint32_t kernel_current_thread_handle(void);
static uint8_t who_bit(void) {
    if (who_n < 0) {
        who_n = 0;
        const char *e = getenv("SAVAGEWIND_FNPROF_WHO");
        if (e) {
            char *p;
            who_main = (uint32_t)strtoul(e, &p, 16);
            while (p && *p && who_n < 16) { if (*p == ':' || *p == ',') p++; who_jobs[who_n++] = (uint32_t)strtoul(p, &p, 16); }
        }
    }
    if (!who_n) return 0;
    uint32_t h = kernel_current_thread_handle();
    if (h == who_main) return 1;
    for (int i = 0; i < who_n; i++) if (h == who_jobs[i]) return 2;
    return 4;
}

// SAVAGEWIND_FNPROF_BT=<hex addr>[,...]: also print the host backtrace (guest functions are f_<addr>) the first time
// each listed function is entered inside the window - who calls it
static uint64_t bt_addrs[256];
static int bt_n = -1;
static _Atomic uint8_t bt_done[256];

uint64_t g_bt_first_x0;

static void bt_check(uint64_t a, uint64_t lr, uint64_t x0) {
    if (bt_n < 0) {
        bt_n = 0;
        const char *e = getenv("SAVAGEWIND_FNPROF_BT");
        for (const char *p = e; p && *p && bt_n < 256;) {
            bt_addrs[bt_n++] = strtoull(p, (char **)&p, 16);
            while (*p == ',' || *p == ' ') p++;
        }
    }
    for (int i = 0; i < bt_n; i++)
        if (bt_addrs[i] == a && !atomic_exchange(&bt_done[i], 1)) {
            if (i == 0) g_bt_first_x0 = x0;
            uint64_t vt = x0 >= 0x7000000000ull && x0 < 0x7800000000ull ? *(uint64_t *)(uintptr_t)x0 : 0;
            uint64_t v100 = vt >= 0x7000000000ull && vt < 0x7800000000ull ? *(uint64_t *)(uintptr_t)(vt + 0x100) : 0;
            uint64_t p8 = x0 >= 0x7000000000ull && x0 < 0x7800000000ull ? *(uint64_t *)(uintptr_t)(x0 + 8) : 0;
            fprintf(stderr, "[fnprof-bt] %llx entered from guest lr %llx x0 %llx (vtable %llx, +0x100 %llx, [x0+8] %llx):\n",
                    (unsigned long long)a, (unsigned long long)lr, (unsigned long long)x0, (unsigned long long)vt,
                    (unsigned long long)v100, (unsigned long long)p8);
            plat_print_backtrace();  // (execinfo is macOS / glibc only)
        }
}

// SAVAGEWIND_FNPROF_ARGS=<hex addr>[@<reg>:<hex offset>:<qwords>][,...]: log x0-x5 and the caller on each entry of
// those functions (the first SAVAGEWIND_FNPROF_ARGS_MAX entries each, default 64) - what a function is called with,
// without a debugger - and what one register points to, at an offset (a field deep in `this`: @0:840:2). Without
// @: SAVAGEWIND_FNPROF_DEREF=<reg>:<qwords> (default 1:4, x1: often a descriptor or out argument), offset 0.
static uint64_t args_addrs[64];
static struct { int reg, n; uint64_t off; } args_deref[64];
static int args_n = -1, args_max = 64, deref_reg = 1, deref_n = 4;
static _Atomic int args_count[64];

static void args_check(uint64_t a, uint64_t lr) {
    if (args_n < 0) {
        args_n = 0;
        const char *e = getenv("SAVAGEWIND_FNPROF_ARGS");
        if (getenv("SAVAGEWIND_FNPROF_ARGS_MAX")) args_max = atoi(getenv("SAVAGEWIND_FNPROF_ARGS_MAX"));
        if (getenv("SAVAGEWIND_FNPROF_DEREF")) sscanf(getenv("SAVAGEWIND_FNPROF_DEREF"), "%d:%d", &deref_reg, &deref_n);
        if (deref_reg < 0 || deref_reg > 30) deref_reg = 1;
        if (deref_n < 0 || deref_n > 64) deref_n = 4;
        for (const char *p = e; p && *p && args_n < 64;) {
            args_deref[args_n].reg = deref_reg, args_deref[args_n].n = deref_n, args_deref[args_n].off = 0;
            args_addrs[args_n] = strtoull(p, (char **)&p, 16);
            if (*p == '@') {
                int reg = 0, n = 0;
                unsigned long long off = 0;
                if (sscanf(p + 1, "%d:%llx:%d", &reg, &off, &n) == 3 && reg >= 0 && reg <= 30 && n >= 0 && n <= 64)
                    args_deref[args_n].reg = reg, args_deref[args_n].n = n, args_deref[args_n].off = off;
                while (*p && *p != ',') p++;
            }
            args_n++;
            while (*p == ',' || *p == ' ') p++;
        }
    }
    for (int i = 0; i < args_n; i++)
        if (args_addrs[i] == a && atomic_fetch_add(&args_count[i], 1) < args_max) {
            Cpu *c = kernel_current_cpu();
            if (!c) return;
            char line[1600];
            int n = snprintf(line, sizeof line, "[fnprof-args] %llx from %llx: %llx %llx %llx %llx %llx %llx [x%d:",
                             (unsigned long long)a, (unsigned long long)lr, (unsigned long long)c->x[0],
                             (unsigned long long)c->x[1], (unsigned long long)c->x[2], (unsigned long long)c->x[3],
                             (unsigned long long)c->x[4], (unsigned long long)c->x[5], args_deref[i].reg);
            if (args_deref[i].off) n += snprintf(line + n, sizeof line - n, "+%llx", (unsigned long long)args_deref[i].off);
            uint64_t p = c->x[args_deref[i].reg] + args_deref[i].off;
            for (int k = 0; k < args_deref[i].n && p >= GUEST_LO && p + 8 * (k + 1) <= GUEST_HI; k++) {
                uint64_t v = ((uint64_t *)(uintptr_t)p)[k];
                n += snprintf(line + n, sizeof line - n, " %llx", (unsigned long long)v);
                const char *s = (const char *)(uintptr_t)v;  // a pointer to text (a name): show it
                int len = 0;
                if (v >= GUEST_LO && v + 64 <= GUEST_HI)
                    while (len < 64 && s[len] >= 0x20 && s[len] < 0x7f) len++;
                if (len >= 4 && (len == 64 || !s[len]) && n < (int)sizeof line - 80)
                    n += snprintf(line + n, sizeof line - n, "=\"%.*s\"", len, s);
            }
            fprintf(stderr, "%s]\n", line);
        }
}

// SAVAGEWIND_FNPROF_GROUPS=<addr of the job runner, 1.9.0: 7000ef62dc>: how big BOTW's job groups are. At a group's
// first chunk (the runner's x0 is the group; its next job +0x94 still 0) its job count (+0x90) is filed under the
// group's vtable, the runner's caller and the first job's vtable; written as build/fnprof_groups.txt (how much work a
// phase can be split into: BOTW-JOBS-NATIVE.md).
// With a second address (SAVAGEWIND_FNPROF_GROUPS=<runner>:<post>, 1.9.0: 7000ef62dc:7000ef6258, the worker's step
// after a group's chunks) each thread's time on a group, from its first chunk to that step, is added up per kind
// (total, longest, per thread kind), and the profile counts nothing else (the light mode: two compares per entry).
static uint64_t grp_runner = ~0ull, grp_post;
static struct {
    _Atomic uint64_t key;
    uint64_t jvt;
    _Atomic uint64_t groups, jobs, max, small, ns, ns_max, ns_main;
} grp[512];
static __thread uint64_t grp_cur, grp_t0;  // (this thread's group in progress and when its first chunk started)
static int grp_slot(uint64_t g, uint64_t *jvt_out) {
    uint64_t vt = *(uint64_t *)(uintptr_t)g;
    uint32_t n = *(uint32_t *)(uintptr_t)(g + 0x90);
    uint64_t jobs = *(uint64_t *)(uintptr_t)(g + 0x88);
    uint64_t jvt = n && jobs ? *(uint64_t *)(uintptr_t)*(uint64_t *)(uintptr_t)jobs : 0;
    uint64_t key = vt ^ (jvt << 3);
    *jvt_out = jvt;
    for (uint32_t h = (uint32_t)(key * 0x9E3779B97F4A7C15ull >> 55), k = 0; k < 512; k++, h = (h + 1) & 511) {
        uint64_t cur = atomic_load_explicit(&grp[h].key, memory_order_relaxed);
        if (cur == key || (cur == 0 && atomic_compare_exchange_strong(&grp[h].key, &cur, key))) return (int)h;
    }
    return -1;
}
static void grp_check(uint64_t g) {
    if (!g) return;
    if (grp_cur != g) grp_cur = g, grp_t0 = plat_now_ns();
    if (*(uint32_t *)(uintptr_t)(g + 0x94)) return;  // (sizes: once a group, at its first chunk)
    uint64_t jvt;
    int h = grp_slot(g, &jvt);
    if (h < 0) return;
    uint32_t n = *(uint32_t *)(uintptr_t)(g + 0x90);
    grp[h].jvt = jvt;
    atomic_fetch_add(&grp[h].groups, 1);
    atomic_fetch_add(&grp[h].jobs, n);
    if (n <= 3) atomic_fetch_add(&grp[h].small, 1);
    for (uint64_t m = atomic_load(&grp[h].max); n > m && !atomic_compare_exchange_weak(&grp[h].max, &m, n);) {}
}
static void grp_done(uint64_t g) {
    if (!g || g != grp_cur) return;
    uint64_t dt = plat_now_ns() - grp_t0, jvt;
    grp_cur = 0;
    int h = grp_slot(g, &jvt);
    if (h < 0) return;
    atomic_fetch_add(&grp[h].ns, dt);
    if (who_bit() == 1) atomic_fetch_add(&grp[h].ns_main, dt);
    for (uint64_t m = atomic_load(&grp[h].ns_max); dt > m && !atomic_compare_exchange_weak(&grp[h].ns_max, &m, dt);) {}
}
static void grp_write(void) {
    FILE *f = fopen("build/fnprof_groups.txt", "w");
    if (!f) return;
    fprintf(f, "# group_vtable first_job_vtable groups jobs avg_jobs max_jobs groups_of_<=3 thread_us_total "
               "thread_us_longest thread_us_main\n");
    for (int i = 0; i < 512; i++)
        if (grp[i].key)
            fprintf(f, "%llx %llx %llu %llu %.1f %llu %llu %.0f %.1f %.0f\n",
                    (unsigned long long)(grp[i].key ^ (grp[i].jvt << 3)), (unsigned long long)grp[i].jvt,
                    (unsigned long long)grp[i].groups, (unsigned long long)grp[i].jobs,
                    grp[i].groups ? (double)grp[i].jobs / grp[i].groups : 0, (unsigned long long)grp[i].max,
                    (unsigned long long)grp[i].small, grp[i].ns / 1e3, grp[i].ns_max / 1e3, grp[i].ns_main / 1e3);
    fclose(f);
}

void fnprof_hit(uint64_t a, uint64_t lr, uint64_t x0) {
    if (grp_runner == ~0ull) {
        const char *e = getenv("SAVAGEWIND_FNPROF_GROUPS");
        char *p = NULL;
        grp_runner = e ? strtoull(e, &p, 16) : 0;
        grp_post = p && *p == ':' ? strtoull(p + 1, NULL, 16) : 0;
    }
    if (grp_runner) {
        if (a == grp_runner) grp_check(x0);
        else if (a == grp_post) grp_done(x0);
        if (grp_post) return;  // (light mode: group times only)
    }
    if (bt_n) bt_check(a, lr, x0);
    if (args_n) args_check(a, lr);
    uint32_t h = (uint32_t)((a >> 2) * 0x9E3779B1u) & (NSLOT - 1);
    for (int n = 0; n < NSLOT; n++, h = (h + 1) & (NSLOT - 1)) {
        uint64_t cur = atomic_load_explicit(&slots[h].addr, memory_order_relaxed);
        if (cur == a || (cur == 0 && atomic_compare_exchange_strong(&slots[h].addr, &cur, a)) || cur == a) {
            atomic_fetch_add_explicit(&slots[h].count, 1, memory_order_relaxed);
            uint8_t w = who_bit();
            if (w && !(atomic_load_explicit(&slots[h].who, memory_order_relaxed) & w)) atomic_fetch_or(&slots[h].who, w);
            return;
        }
    }
}

static int cmp(const void *x, const void *y) {
    uint64_t a = ((const uint64_t *)x)[1], b = ((const uint64_t *)y)[1];
    return a < b ? 1 : a > b ? -1 : 0;
}

void fnprof_frame(uint64_t frame) {
    static long from = -2, to;
    if (from == -2) {
        from = -1;
        const char *e = getenv("SAVAGEWIND_FNPROF");
        if (e && sscanf(e, "%ld:%ld", &from, &to) != 2) from = -1;
    }
    if (from < 0) return;
    if ((long)frame == from) { g_fnprof = 1; LOG("[fnprof] on at frame %llu\n", (unsigned long long)frame); }
    if ((long)frame != to) return;
    g_fnprof = 0;
    uint64_t (*rows)[3] = malloc(sizeof(uint64_t[3]) * NSLOT);
    int n = 0;
    for (int i = 0; i < NSLOT; i++)
        if (slots[i].addr) { rows[n][0] = slots[i].addr; rows[n][1] = slots[i].count; rows[n][2] = slots[i].who; n++; }
    qsort(rows, n, sizeof rows[0], cmp);
    const char *out = getenv("SAVAGEWIND_FNPROF_OUT") ? getenv("SAVAGEWIND_FNPROF_OUT") : "build/fnprof.txt";  // (parallel runs)
    FILE *f = fopen(out, "w");
    for (int i = 0; f && i < n; i++) fprintf(f, "%llx %llu %llu\n", (unsigned long long)rows[i][0], (unsigned long long)rows[i][1], (unsigned long long)rows[i][2]);
    if (f) fclose(f);
    free(rows);
    if (grp_runner && grp_runner != ~0ull) grp_write();
    LOG("[fnprof] %d functions, written %s (frames %ld-%ld)\n", n, out, from, to);
    from = -1;
}

// SAVAGEWIND_FNPROF=0:<to>: profile from process start (threads the game starts before the first frame)
__attribute__((constructor)) static void fnprof_early(void) {
    const char *e = getenv("SAVAGEWIND_FNPROF");
    if (e && e[0] == '0' && e[1] == ':') g_fnprof = 1;
}
