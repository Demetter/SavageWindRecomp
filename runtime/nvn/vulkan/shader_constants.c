#include "shader_constants.h"
#include <ctype.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static bool identifier(char c) { return isalnum((unsigned char)c) || c == '_'; }

char *vk_shader_literal_constants(const char *source, const uint8_t *data, size_t size) {
    if (!source || !data || !size || (size & 3) || size > 65536) return NULL;
    const char *decl = strstr(source, "binding = 1) uniform CB1 { uvec4 cb1[");
    if (!decl) return NULL;
    const char *line = decl;
    while (line > source && line[-1] != '\n') line--;
    unsigned set, count;
    int consumed = 0;
    if (sscanf(line, "layout(std140, set = %u, binding = 1) uniform CB1 { uvec4 cb1[%u]; };%n", &set, &count, &consumed) != 2 ||
        !consumed || set > 1 || !count || count > 4096 || (line[consumed] && line[consumed] != '\n' && line[consumed] != '\r')) return NULL;
    const char *after = line + consumed;
    if (*after == '\r') after++;
    if (*after == '\n') after++;
    size_t len = strlen(source);
    if (len > (SIZE_MAX - 1) / 2) return NULL;
    char *out = malloc(len * 2 + 1), *d = out;
    if (!out) return NULL;
    unsigned replaced = 0;
    for (const char *p = source; *p;) {
        if (p == line) { p = after; continue; }
        if (!strncmp(p, "cb1", 3) && (p == source || !identifier(p[-1])) && !identifier(p[3])) {
            // Every c1 reference must be one exact, in-bounds scalar load. A
            // cbr(cb1, ...) or unrecognized use rejects the entire specialization.
            if (p[3] != '[' || !isdigit((unsigned char)p[4])) goto fallback;
            char *end;
            unsigned long row = strtoul(p + 4, &end, 10);
            if (row >= count || end[0] != ']' || end[1] != '[' ||
                end[2] < '0' || end[2] > '3' || end[3] != ']') goto fallback;
            size_t offset = (size_t)row * 16 + (unsigned)(end[2] - '0') * 4;
            if (offset > size || size - offset < 4) goto fallback;
            uint32_t bits;
            memcpy(&bits, data + offset, sizeof bits);
            d += sprintf(d, "0x%08xu", (unsigned)bits);
            p = end + 4;
            replaced++;
        } else *d++ = *p++;
    }
    *d = 0;
    if (replaced) return out;
fallback:
    free(out);
    return NULL;
}
