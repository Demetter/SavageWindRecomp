#include "native_draw_constants.h"
#include <ctype.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { char *p; size_t n, capacity; bool failed; } Text;
static void append(Text *t, const char *p, size_t n) {
    if (t->failed || n > SIZE_MAX - t->n - 1) { t->failed = true; return; }
    size_t need = t->n + n + 1;
    if (need > t->capacity) {
        size_t capacity = need > SIZE_MAX / 2 ? need : need * 2;
        char *q = realloc(t->p, capacity);
        if (!q) { t->failed = true; return; }
        t->p = q; t->capacity = capacity;
    }
    memcpy(t->p + t->n, p, n); t->n += n; t->p[t->n] = 0;
}
static void format(Text *t, const char *fmt, ...) {
    char buf[2048]; va_list args; va_start(args, fmt);
    int n = vsnprintf(buf, sizeof buf, fmt, args); va_end(args);
    if (n < 0 || (size_t)n >= sizeof buf) { t->failed = true; return; }
    append(t, buf, (size_t)n);
}
static bool identifier(char c) { return isalnum((unsigned char)c) || c == '_'; }

char *vk_shader_native_draw_constants(const char *source, unsigned stage,
                                     const char *draw_index, uint32_t *mask) {
    if (mask) *mask = 0;
    if (!source || stage > 1 || !draw_index || !*draw_index ||
        strncmp(source, "#version ", 9) || strstr(source, "SWNative") || strstr(source, "sw_native_")) return NULL;
    const char *version_end = strchr(source, '\n');
    if (!version_end) return NULL;
    version_end++;
    unsigned counts[18] = {0};
    const char *begin[18] = {0}, *end[18] = {0}, *first = NULL;
    uint32_t used = 0;
    for (const char *p = source; *p;) {
        const char *next = strchr(p, '\n'); if (!next) next = p + strlen(p); else next++;
        unsigned set, binding, block, name, count; int n = 0;
        if (sscanf(p, "layout(std140, set = %u, binding = %u) uniform CB%u { uvec4 cb%u[%u]; };%n",
                   &set, &binding, &block, &name, &count, &n) == 5 && n > 0) {
            char canonical[160];
            int canonical_size = snprintf(canonical, sizeof canonical,
                "layout(std140, set = %u, binding = %u) uniform CB%u { uvec4 cb%u[%u]; };",
                set, binding, block, name, count);
            if (set != stage || binding != block || binding != name || name >= 18 ||
                !count || count > 4096 || counts[name] || canonical_size != n ||
                memcmp(p, canonical, (size_t)n) || (p[n] && p[n] != '\r' && p[n] != '\n')) return NULL;
            counts[name] = count; begin[name] = p; end[name] = next; used |= 1u << name;
            if (!first) first = p;
        }
        p = next;
    }
    if (!used) return NULL;
    Text out = {0};
    append(&out, source, (size_t)(version_end - source));
    format(&out, "#extension GL_EXT_buffer_reference2 : require\n#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require\n");
    for (const char *p = version_end; *p && !out.failed;) {
        if (p == first) {
            format(&out,
                "layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer SWNativeWords { uint words[]; };\n"
                "struct SWNativeConstant { uint64_t address; uint words; uint reserved; };\n"
                "struct SWNativeDraw { SWNativeConstant constants[36]; };\n"
                "layout(std430, set = 2, binding = 0) readonly buffer SWNativeTable { SWNativeDraw draws[]; } sw_native_table;\n");
            for (unsigned k = 0; k < 18; k++) if (counts[k]) {
                // Raw addresses have no descriptor robustness. Guard every word
                // before dereferencing; this also handles partial vec4 ranges.
                format(&out,
                    "uvec4 sw_native_cb%u(uint row) {\n"
                    " SWNativeConstant c = sw_native_table.draws[%s].constants[%u];\n"
                    " if (c.address == uint64_t(0) || row >= %uu) return uvec4(0);\n"
                    " uint word = row * 4u; uvec4 v = uvec4(0);\n"
                    " for (uint i=0u;i<4u;i++) if (word+i < c.words) v[i] = SWNativeWords(c.address).words[word+i];\n"
                    " return v; }\n"
                    "uint sw_native_cbr%u(uint off) { return sw_native_cb%u(min(off >> 4u, %uu))[(off >> 2u) & 3u]; }\n",
                    k, draw_index, stage * 18 + k, counts[k], k, k, counts[k] - 1);
            }
        }
        bool declaration = false;
        for (unsigned k = 0; k < 18; k++) if (p == begin[k]) { p = end[k]; declaration = true; break; }
        if (declaration) continue;
        // Generated PC comments can contain hex offsets such as "cb0". They
        // are not loads and must neither be rewritten nor reject a valid shader.
        if (p[0] == '/' && (p[1] == '/' || p[1] == '*')) {
            const char *stop = p[1] == '/' ? strchr(p, '\n') : strstr(p + 2, "*/");
            if (!stop && p[1] == '*') goto fail;
            if (!stop) stop = p + strlen(p);
            else stop += p[1] == '*' ? 2 : 1;
            append(&out, p, (size_t)(stop - p)); p = stop; continue;
        }
        if ((p == source || !identifier(p[-1])) && !strncmp(p, "cbr(cb", 6) && isdigit((unsigned char)p[6])) {
            char *q; unsigned long k = strtoul(p + 6, &q, 10);
            if (q == p + 6 || k >= 18 || !counts[k] || *q != ',') goto fail;
            format(&out, "sw_native_cbr%lu(", k); p = q + 1; continue;
        }
        if ((p == source || !identifier(p[-1])) && p[0] == 'c' && p[1] == 'b' && isdigit((unsigned char)p[2])) {
            char *q; unsigned long k = strtoul(p + 2, &q, 10);
            if (identifier(*q)) { append(&out, p, (size_t)(q - p)); p = q; continue; }
            if (k >= 18 || !counts[k] || *q != '[' || !isdigit((unsigned char)q[1])) goto fail;
            char *r; unsigned long row = strtoul(q + 1, &r, 10);
            if (*r != ']' || row >= counts[k]) goto fail;
            format(&out, "sw_native_cb%lu(%luu)", k, row); p = r + 1; continue;
        }
        append(&out, p++, 1);
    }
    if (out.failed) goto fail;
    if (mask) *mask = used;
    return out.p;
fail:
    free(out.p); return NULL;
}

char *vk_shader_native_draw_graphics(const char *source, unsigned stage,
                                    unsigned location, uint32_t *mask) {
    if (mask) *mask = 0;
    if (!source || stage > 1 || location >= 32 || strncmp(source, "#version ", 9) ||
        strstr(source, "sw_batch_") || strstr(source, "push_constant") || strstr(source, "gl_DrawID")) return NULL;
    char declaration[64];
    // The caller must choose a shared free VS/FS location AND account for all
    // builtin components in maxVertexOutputComponents. A free location 31 can
    // exceed a 128-component device limit once gl_Position/PointSize are counted.
    snprintf(declaration, sizeof declaration, "layout(location = %u)", location);
    // Also rejects an unrelated vertex input at this location, conservatively.
    if (strstr(source, declaration)) return NULL;
    const char *main = strstr(source, "void main() {\n");
    if (!main || strstr(main + 1, "void main()")) return NULL;
    const char *header = strchr(source, '\n'); if (!header) return NULL; header++;
    while (!strncmp(header, "#extension ", 11)) {
        const char *next = strchr(header, '\n'); if (!next) return NULL; header = next + 1;
    }
    Text text = {0};
    append(&text, source, (size_t)(header - source));
    format(&text, "layout(location = %u) flat %s uint sw_batch_draw_id;\n", location, stage ? "in" : "out");
    if (!stage) format(&text, "layout(push_constant) uniform SWBatchRoot { uint first; } sw_batch_root;\n");
    const char *body = main + strlen("void main() {\n");
    append(&text, header, (size_t)(body - header));
    if (!stage) format(&text, " sw_batch_draw_id = uint(gl_DrawID) + sw_batch_root.first;\n");
    append(&text, body, strlen(body));
    if (text.failed) { free(text.p); return NULL; }
    if (!strstr(source, "uniform CB")) return text.p; // no constant buffers, still propagate the draw ID
    char *out = vk_shader_native_draw_constants(text.p, stage, "sw_batch_draw_id", mask);
    free(text.p); return out;
}
