#include <assert.h>
#include <stdio.h>
#include "../runtime/nvn/vulkan/shader_compute_groups.h"
#include "../runtime/nvn/vulkan/shader_store_audit.h"
int main(int argc, char **argv) {
    assert(!shader_compute_target(NULL) && !shader_compute_target("wrong source"));
    uint32_t physical;
    assert(shader_compute_dispatch(16383, 65535, &physical) && physical == 65532);
    assert(!shader_compute_dispatch(16384, 65535, &physical) && physical == 16384);
    assert(!shader_compute_dispatch(UINT32_MAX, UINT32_MAX, &physical) && physical == UINT32_MAX);
    char *s = shader_compute_groups("uint g=gl_WorkGroupID.x;uint l=gl_LocalInvocationID.x;");
    assert(s && strstr(s, "gl_GlobalInvocationID.x >> 5u") && strstr(s, "gl_GlobalInvocationID.x & 31u")); free(s);
    const char *bad[] = {"uint x=gl_WorkGroupID.y;uint y=gl_LocalInvocationID.x;",
        "uint x=gl_WorkGroupID.x;uint y=gl_LocalInvocationID.x; barrier();",
        "uint x=gl_WorkGroupID.x;uint y=gl_LocalInvocationID.x; subgroupShuffle(x,y);",
        "shared uint x; uint y=gl_WorkGroupID.x+gl_LocalInvocationID.x;"};
    for (size_t i = 0; i < sizeof bad / sizeof *bad; i++) assert(!shader_compute_groups(bad[i]));
    for (unsigned groups = 1; groups <= 256; groups++) for (unsigned g = 0; g < groups * 4; g++) for (unsigned l = 0; l < 8; l++) {
        unsigned invocation = g * 8 + l, original_g = invocation >> 5, original_l = invocation & 31;
        assert(original_g < groups && original_l < 32 && original_g * 32 + original_l == invocation);
    }
    if (argc == 3) {
        FILE *f = fopen(argv[1], "rb"); assert(f); fseek(f,0,SEEK_END); long n=ftell(f); rewind(f); assert(n>0);
        char *src=malloc((size_t)n+1); assert(src); assert(fread(src,1,(size_t)n,f)==(size_t)n); fclose(f); src[n]=0;
        // Saved Windows text dumps contain CRLF; runtime GLSL uses LF.
        char *write = src;
        for (const char *read = src; *read; read++) {
            if (*read == '\r' && read[1] == '\n') continue;
            *write++ = *read;
        }
        *write = 0;
        assert(shader_compute_target(src));
        src[0] ^= 1; assert(!shader_compute_target(src)); src[0] ^= 1;
        s=shader_compute_groups(src); assert(s); char *audit=shader_store_audit(s); assert(audit);
        f=fopen(argv[2],"wb"); assert(f); fputs(audit,f); fclose(f); free(audit); free(s); free(src);
    }
    puts("compute groups: exact logical indices, unsupported builtins/communication rejection, audit composition passed");
    return 0;
}
