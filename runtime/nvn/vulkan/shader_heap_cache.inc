// Compute-only, exact word reads/stores through a device-local working set.
// Unsupported syntax/atomics retain the original shader. The imported heap remains
// authoritative: stores update both copies; an unmapped page reads the heap.
// Included separately so the rewrite can be tested without the renderer.
#include <ctype.h>

static char *shader_heap_cache(const char *src, unsigned pages, bool verify) {
    if (!src || !pages || strstr(src, "atomic") || strstr(src, "binding = 20") || strstr(src, "binding = 21")) return NULL;
    const char *main = strstr(src, "void main()");
    if (!main || !strstr(main, "gmem_.w[")) return NULL;
    size_t cap = strlen(src) * 2 + 4096, used = 0;
    char *out = malloc(cap);
    if (!out) return NULL;
    size_t prefix = (size_t)(main - src);
    memcpy(out, src, prefix); used = prefix;
    used += (size_t)snprintf(out + used, cap - used,
        "layout(std430, set=0, binding=20) buffer HeapCache { uint w[]; } hc_;\n"
        "layout(std430, set=0, binding=21) buffer HeapMap { uint w[]; } hm_;\n"
        "uint hc_slot(uint i) { uint p=i>>14u; return p<%uu ? hm_.w[p] : 0u; }\n"
        "uint hc_read(uint i) { uint s=hc_slot(i); uint v=s!=0u ? hc_.w[((s-1u)<<14u)+(i&16383u)] : gmem_.w[i];\n",
        pages);
    if (verify) used += (size_t)snprintf(out + used, cap - used,
        " if(hm_.w[%uu]!=0u) { atomicAdd(hm_.w[%uu],1u); if(s!=0u) { atomicAdd(hm_.w[%uu],1u); if(v!=gmem_.w[i]) atomicAdd(hm_.w[%uu],1u); } }\n",
        pages + 2, pages + 3, pages + 1, pages);
    used += (size_t)snprintf(out + used, cap - used, " return v; }\n");
    // A readonly heap declaration means there are no stores; do not introduce an
    // unused function that writes it (GLSL rejects the definition nonetheless).
    bool readonly = strstr(src, "readonly buffer GMem") != NULL;
    if (!readonly) used += (size_t)snprintf(out + used, cap - used,
        "void hc_write(uint i,uint v) { uint s=hc_slot(i); gmem_.w[i]=v; if(s!=0u) hc_.w[((s-1u)<<14u)+(i&16383u)]=v; }\n");
    const char *p = main;
    while (*p) {
        const char *a = strstr(p, "gmem_.w[");
        if (!a) { size_t n = strlen(p); memcpy(out + used, p, n + 1); return out; }
        size_t n = (size_t)(a - p);
        memcpy(out + used, p, n); used += n;
        const char *idx = a + 8, *end = idx;
        int depth = 1;
        while (*end && depth) { if (*end == '[') depth++; if (*end == ']') depth--; if (depth) end++; }
        if (depth || end - idx > 256) break;
        const char *q = end + 1;
        while (isspace((unsigned char)*q)) q++;
        if (*q == '=' && q[1] != '=') {
            if (readonly) break;
            const char *val = q + 1, *semi = strchr(val, ';');
            if (!semi || semi - val > 64 || strstr(val, "gmem_.w[") == val) break;
            // Translator stores are a register or zero, never compound expressions.
            while (isspace((unsigned char)*val)) val++;
            const char *v = val;
            if (*v == 'r' && isdigit((unsigned char)v[1])) { v++; while (isdigit((unsigned char)*v)) v++; }
            else if (!strncmp(v, "0u", 2)) v += 2;
            else break;
            while (isspace((unsigned char)*v)) v++;
            if (v != semi) break;
            used += (size_t)snprintf(out + used, cap - used, "hc_write(%.*s,%.*s);", (int)(end-idx), idx, (int)(semi-val), val);
            p = semi + 1;
        } else {
            if (*q == '+' || *q == '-' || *q == '*' || *q == '/' || *q == '&' || *q == '|') {
                if (q[1] == '=' || q[1] == *q) break; // reject compound writes
            }
            used += (size_t)snprintf(out + used, cap - used, "hc_read(%.*s)", (int)(end-idx), idx);
            p = end + 1;
        }
        if (used + strlen(p) + 1024 >= cap) break;
    }
    free(out);
    return NULL;
}
