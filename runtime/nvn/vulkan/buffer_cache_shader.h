// A1 diagnostic routing. Every covered write lands only in owned GPU storage.
// Uncovered accesses retain the original heap semantics and set a coverage bit;
// a zero miss count is retrospective evidence, never an early-release proof.
#pragma once
#include <ctype.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BC_SHADER_COUNT 64
#define BC_AUDIT_WORDS 32
// Exact-source proof for c3e522014faa8ebd: all 13 loads are
// sbuf[0].address + 64*gl_InstanceIndex + {0..44,60}; no loaded value
// contributes to another global address. Runtime additionally bounds the
// instance interval and excludes 32-bit stride arithmetic overflow.
static bool buffer_cache_graphics_proved(const char *src, int stage) {
    uint64_t h=UINT64_C(14695981039346656037);
    if (!src || stage!=0) return false;
    for(const unsigned char *p=(const unsigned char *)src;*p;p++) if(*p!='\r') h=(h^*p)*UINT64_C(1099511628211);
    return h==UINT64_C(0x4a059cead83af2ce);
}
static char *buffer_cache_shader_stage(const char *src, int stage) {
    if (!src || !strstr(src, "gmem_.w[") || strstr(src, "a1_read(") ||
        strstr(src, "atomic") || strstr(src, "binding = 23") || strstr(src, "binding=23")) return NULL;
    const char *main = strstr(src, "void main()");
    if (!main) return NULL;
    size_t cap = strlen(src) * 3 + 8192, used = (size_t)(main - src);
    char *out = malloc(cap); if (!out) return NULL;
    memcpy(out, src, used);
    used += (size_t)snprintf(out + used, cap - used,
        "// A1b-owned-buffer-v2; separate source/cache identity\n"
        "layout(std430,set=%d,binding=23) buffer A1Owned { uint w[]; } a1_;\n"
        "layout(std430,set=%d,binding=24) buffer A1Dirty { uint w[]; } a1d_;\n"
        "layout(std430,set=%d,binding=25) readonly buffer A1Ranges { uint w[]; } a1r_;\n",
        stage==1,stage==1,stage==1);
    used += (size_t)snprintf(out+used,cap-used,
        "bool a1_has(uint i) { uint lo=0u, hi=a1r_.w[0]; while(lo<hi) { uint m=(lo+hi)>>1u;"
        " if(a1r_.w[5u+2u*m]<=i) lo=m+1u; else hi=m; }"
        " return lo<a1r_.w[0] && a1r_.w[4u+2u*lo]<=i; }\n");
    if(stage==0 || stage==1) used+=(size_t)snprintf(out+used,cap-used,
        "layout(std430,set=%d,binding=26) buffer A1Audit {uint w[];} a1a_;\n"
        "void a1_observe(uint i,bool wr){if(a1r_.w[3]==0u)return;uint b=a1r_.w[2]*32u;"
        "uint k=atomicAdd(a1a_.w[b+(wr?1u:0u)],1u);atomicMax(a1a_.w[b+2u],~i);atomicMax(a1a_.w[b+3u],i);"
        "if(!a1_has(i))atomicAdd(a1a_.w[b+4u],1u);if(!wr&&k<16u)a1a_.w[b+8u+k]=i;}\n",stage==1);
    else used+=(size_t)snprintf(out+used,cap-used,"void a1_observe(uint i,bool wr){}\n");
    used += (size_t)snprintf(out+used,cap-used,
        "uint a1_read(uint i) { a1_observe(i,false); if(a1r_.w[1]!=1u) return gmem_.w[i];"
        " if(a1_has(i)) return a1_.w[i];"
        " atomicOr(a1d_.w[0],1u); atomicAdd(a1d_.w[2],1u); return gmem_.w[i]; }\n");
    bool readonly = strstr(src, "readonly buffer GMem") != NULL;
    if (!readonly) used += (size_t)snprintf(out + used, cap - used,
        "void a1_write(uint i,uint v) { a1_observe(i,true); if(a1r_.w[1]!=1u) {gmem_.w[i]=v;return;}"
        " if(a1_has(i)) { a1_.w[i]=v; uint bit=1u<<(i&31u);"
        " uint old=atomicOr(a1d_.w[4u+(i>>5u)],bit);"
        " if((old&bit)==0u) atomicAdd(a1d_.w[1],4u); }"
        " else {atomicOr(a1d_.w[0],2u);atomicAdd(a1d_.w[3],1u);gmem_.w[i]=v;} }\n");
    const char *p = main;
    while (*p) {
        const char *a = strstr(p, "gmem_.w[");
        if (!a) { memcpy(out + used, p, strlen(p) + 1); return out; }
        size_t n = (size_t)(a - p); memcpy(out + used, p, n); used += n;
        const char *idx = a + 8, *end = idx; int depth = 1;
        while (*end && depth) { if (*end == '[') depth++; if (*end == ']') depth--; if (depth) end++; }
        if (depth || end - idx > 256) break;
        const char *q = end + 1; while (isspace((unsigned char)*q)) q++;
        if (*q == '=' && q[1] != '=') {
            if (readonly) break;
            const char *value = q + 1, *semi = strchr(value, ';');
            if (!semi || semi - value > 64) break;
            while (isspace((unsigned char)*value)) value++;
            const char *v = value;
            if (*v == 'r' && isdigit((unsigned char)v[1])) { v++; while (isdigit((unsigned char)*v)) v++; }
            else if (!strncmp(v,"0u",2)) v+=2; else break;
            while (isspace((unsigned char)*v)) v++;
            if (v != semi) break;
            used += (size_t)snprintf(out+used,cap-used,"a1_write(%.*s,%.*s);",(int)(end-idx),idx,(int)(semi-value),value);
            p=semi+1;
        } else {
            if (strchr("+-*/&|",*q) && *q && (q[1]=='=' || q[1]==*q)) break;
            used += (size_t)snprintf(out+used,cap-used,"a1_read(%.*s)",(int)(end-idx),idx); p=end+1;
        }
        if (used + strlen(p) + 1024 >= cap) break;
    }
    free(out); return NULL;
}
static char *buffer_cache_shader(const char *src) { return buffer_cache_shader_stage(src,5); }
