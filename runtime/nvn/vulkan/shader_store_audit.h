// Diagnostic transformation for compute kernels without cross-invocation memory
// communication. Redirect stores into a per-invocation journal; both compiler
// variants can then run against identical input without modifying guest memory.
// Reads forward this invocation's previous stores. Read addresses are journaled
// too, so the host can reject cross-invocation read/write or write/write overlap.
// Unsupported syntax and shared/atomic/image operations fail closed. This is an
// output-equivalence probe, never the production shader.
#pragma once
#include <ctype.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define STORE_AUDIT_BINDING 22
#define STORE_AUDIT_MAX 32u
#define STORE_AUDIT_READS 512u
#define STORE_AUDIT_READ_BASE (2u + 2u * STORE_AUDIT_MAX)
#define STORE_AUDIT_WORDS (STORE_AUDIT_READ_BASE + STORE_AUDIT_READS)

static char *shader_store_audit(const char *src) {
    if (!src || strstr(src, "atomic") || strstr(src, "shared ") || strstr(src, "imageStore") || strstr(src, "barrier(") || strstr(src, "Barrier(") ||
        strstr(src, "audit_") || strstr(src, "binding = 22") || strstr(src, "binding=22")) return NULL;
    const char *main = strstr(src, "void main()");
    if (!main || !strstr(main, "gmem_.w[")) return NULL;
    size_t len = strlen(src), cap = len * 2 + 4096, used = (size_t)(main - src);
    char *out = malloc(cap);
    if (!out) return NULL;
    memcpy(out, src, used);
    used += (size_t)snprintf(out + used, cap - used,
        "layout(std430,set=0,binding=22) buffer AuditOutput { uint w[]; } audit_;\n"
        "uint audit_cursor=0u, audit_reads=0u;\n"
        "uint audit_addresses[%u],audit_values[%u];\n"
        "uint audit_base() {\n"
        " uvec3 extent=gl_NumWorkGroups*gl_WorkGroupSize;\n"
        " uint invocation=gl_GlobalInvocationID.x+extent.x*(gl_GlobalInvocationID.y+extent.y*gl_GlobalInvocationID.z);\n"
        " return invocation*%uu; }\n"
        "void audit_store(uint address,uint value) { uint base=audit_base();\n"
        " if(audit_cursor<%uu) { audit_addresses[audit_cursor]=address; audit_values[audit_cursor]=value;"
        " audit_.w[base+2u+audit_cursor*2u]=address; audit_.w[base+3u+audit_cursor*2u]=value; }\n"
        " audit_cursor++; audit_.w[base]=audit_cursor;\n}\n"
        "uint audit_read(uint address) { uint base=audit_base();\n"
        " if(audit_reads<%uu) audit_.w[base+%uu+audit_reads]=address;\n"
        " audit_reads++; audit_.w[base+1u]=audit_reads;\n"
        " for(uint i=min(audit_cursor,%uu);i>0u;i--) if(audit_addresses[i-1u]==address) return audit_values[i-1u];\n"
        " return gmem_.w[address]; }\n", STORE_AUDIT_MAX, STORE_AUDIT_MAX, STORE_AUDIT_WORDS, STORE_AUDIT_MAX,
        STORE_AUDIT_READS, STORE_AUDIT_READ_BASE, STORE_AUDIT_MAX);
    const char *p = main;
    bool stored = false;
    while (*p) {
        const char *a = strstr(p, "gmem_.w[");
        if (!a) {
            if (!stored) break;
            memcpy(out + used, p, strlen(p) + 1);
            return out;
        }
        size_t n = (size_t)(a - p);
        memcpy(out + used, p, n); used += n;
        const char *idx = a + 8, *end = idx;
        int depth = 1;
        while (*end && depth) { if (*end == '[') depth++; if (*end == ']') depth--; if (depth) end++; }
        if (depth || end - idx > 256) break;
        const char *q = end + 1;
        while (isspace((unsigned char)*q)) q++;
        if (*q == '=' && q[1] != '=') {
            const char *val = q + 1, *semi = strchr(val, ';');
            if (!semi || semi - val > 64) break;
            while (isspace((unsigned char)*val)) val++;
            const char *v = val;
            if (*v == 'r' && isdigit((unsigned char)v[1])) { v++; while (isdigit((unsigned char)*v)) v++; }
            else if (!strncmp(v, "0u", 2)) v += 2;
            else break;
            while (isspace((unsigned char)*v)) v++;
            if (v != semi) break;
            used += (size_t)snprintf(out + used, cap - used, "audit_store(%.*s,%.*s);",
                (int)(end - idx), idx, (int)(semi - val), val);
            p = semi + 1;
            stored = true;
        } else {
            if (((*q == '+' || *q == '-' || *q == '*' || *q == '/' || *q == '&' || *q == '|') &&
                           (q[1] == '=' || q[1] == *q))) break;
            used += (size_t)snprintf(out + used, cap - used, "audit_read(%.*s)", (int)(end - idx), idx);
            p = end + 1;
        }
        if (used + strlen(p) + 1024 >= cap) break;
    }
    free(out);
    return NULL;
}

typedef struct { uint64_t stores, reads, differences; } StoreAuditStats;
typedef struct { uint32_t address, invocation; } StoreAuditWriter;
static int store_audit_order(const void *x, const void *y) {
    const StoreAuditWriter *a = x, *b = y;
    if (a->address != b->address) return a->address < b->address ? -1 : 1;
    return a->invocation < b->invocation ? -1 : a->invocation != b->invocation;
}
// 0: exact output/addresses and independent invocations. Negative: an invalid
// probe (overflow/allocation/zero coverage/dependencies). Positive: mismatch.
static int store_audit_check(const uint32_t *a, const uint32_t *b, size_t invocations, StoreAuditStats *stats) {
    *stats = (StoreAuditStats){0};
    if (!invocations || invocations > 65536) return -1;
    for (size_t i = 0; i < invocations; i++) {
        size_t base = i * STORE_AUDIT_WORDS;
        if (a[base] > STORE_AUDIT_MAX || b[base] > STORE_AUDIT_MAX ||
            a[base+1] > STORE_AUDIT_READS || b[base+1] > STORE_AUDIT_READS) return -1;
        stats->stores += a[base]; stats->reads += a[base+1];
        for (unsigned k = 0; k < STORE_AUDIT_WORDS; k++) stats->differences += a[base+k] != b[base+k];
    }
    if (stats->differences) return 1;
    if (!stats->stores) return -2;
    StoreAuditWriter *writers = malloc((size_t)stats->stores * sizeof *writers);
    if (!writers) return -3;
    size_t n = 0;
    for (size_t i = 0; i < invocations; i++) {
        size_t base = i * STORE_AUDIT_WORDS;
        for (unsigned k = 0; k < a[base]; k++) writers[n++] = (StoreAuditWriter){a[base+2+k*2], (uint32_t)i};
    }
    qsort(writers, n, sizeof *writers, store_audit_order);
    for (size_t i = 1; i < n; i++) if (writers[i].address == writers[i-1].address && writers[i].invocation != writers[i-1].invocation) {
        free(writers); return -4;
    }
    for (size_t i = 0; i < invocations; i++) {
        size_t base = i * STORE_AUDIT_WORDS;
        for (unsigned k = 0; k < a[base+1]; k++) {
            uint32_t address = a[base+STORE_AUDIT_READ_BASE+k];
            size_t lo = 0, hi = n;
            while (lo < hi) { size_t mid = lo + (hi-lo)/2; if (writers[mid].address < address) lo = mid + 1; else hi = mid; }
            if (lo < n && writers[lo].address == address && writers[lo].invocation != i) { free(writers); return -5; }
        }
    }
    free(writers); return 0;
}
