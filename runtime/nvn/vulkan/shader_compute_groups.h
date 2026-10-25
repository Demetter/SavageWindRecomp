// Preserve the console kernel's logical 32-thread group/local IDs while using
// smaller physical groups. Only independent 1D kernels without subgroup/shared
// communication are eligible; runtime output audits additionally check memory
// dependencies. The caller multiplies group count by 4 and uses local size 8.
#pragma once
#include <ctype.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// Identity of the exact GLSL used by the compute store/read-journal differential.
// Translation cache IDs also include the translator build and cache policy, so
// they must not select this experiment: unrelated source edits invalidate them.
// Changed shader source fails closed and must be audited before adding a new ID.
static bool shader_compute_target(const char *src) {
    if (!src || strlen(src) != 133059) return false;
    uint64_t h = UINT64_C(0xcbf29ce484222325);
    for (const unsigned char *p = (const unsigned char *)src; *p; p++)
        h = (h ^ *p) * UINT64_C(0x100000001b3);
    return h == UINT64_C(0xcc78b3e8af9676a5);
}

static bool shader_compute_dispatch(uint32_t logical, uint32_t limit, uint32_t *physical) {
    *physical = logical;
    if (logical > limit / 4) return false;
    *physical = logical * 4;
    return true;
}

static char *shader_compute_groups(const char *src) {
    if (!src || strstr(src, "shared ") || strstr(src, "atomic") || strstr(src, "barrier(") ||
        strstr(src, "Barrier(") || strstr(src, "imageStore") || strstr(src, "gl_Subgroup")) return NULL;
    // Extension declarations can mention subgroup features without using them.
    for (const char *p = src; (p = strstr(p, "subgroup")); p += 8) {
        const char *end = p;
        while (isalnum((unsigned char)*end) || *end == '_') end++;
        while (isspace((unsigned char)*end)) end++;
        if (*end == '(') return NULL;
    }
    const char *names[2] = {"gl_WorkGroupID.x", "gl_LocalInvocationID.x"};
    const char *values[2] = {"(gl_GlobalInvocationID.x >> 5u)", "(gl_GlobalInvocationID.x & 31u)"};
    size_t cap = strlen(src) * 3 + 1, used = 0;
    char *out = malloc(cap);
    if (!out) return NULL;
    unsigned counts[2] = {0};
    for (const char *p = src; *p;) {
        if (!strncmp(p, "gl_", 3)) {
            int which = -1;
            for (int i = 0; i < 2; i++) if (!strncmp(p, names[i], strlen(names[i])) &&
                !(isalnum((unsigned char)p[strlen(names[i])]) || p[strlen(names[i])] == '_')) which = i;
            if (which < 0) { free(out); return NULL; }
            size_t n = strlen(values[which]); memcpy(out + used, values[which], n); used += n;
            p += strlen(names[which]); counts[which]++;
        } else out[used++] = *p++;
    }
    out[used] = 0;
    if (counts[0] && counts[1]) return out;
    free(out); return NULL;
}
