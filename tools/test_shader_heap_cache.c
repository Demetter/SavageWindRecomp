#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../runtime/nvn/vulkan/shader_heap_cache.inc"
int main(int argc, char **argv) {
    const char *src = "#version 460\nlayout(std430,set=0,binding=17) buffer GMem { uint w[]; } gmem_;\n"
        "void main() { uint r0 = gmem_.w[uint(8) + 1u]; if(true) gmem_.w[2u] = r0; }\n";
    char *s = shader_heap_cache(src, 53248, false);
    assert(s && strstr(s, "hc_read(uint(8) + 1u)") && strstr(s, "hc_write(2u,r0)"));
    assert(!strstr(strstr(s, "void main()"), "gmem_.w[")); free(s);
    s = shader_heap_cache(src, 53248, true);
    assert(s && strstr(s, "atomicAdd(hm_.w[53249u],1u)") && strstr(s, "if(v!=gmem_.w[i])")); free(s);
    s = shader_heap_cache("readonly buffer GMem { uint w[]; } gmem_;\nvoid main() { uint r0=gmem_.w[1]; }", 8, false);
    assert(s && !strstr(s, "hc_write")); free(s);
    assert(!shader_heap_cache("void main() { atomicAdd(gmem_.w[0],1); }", 8, false));
    assert(!shader_heap_cache("void main() { gmem_.w[0] += 1; }", 8, false));
    assert(!shader_heap_cache("void main() { gmem_.w[0] = f(); }", 8, false));
    assert(!shader_heap_cache("void main() { uint r0 = gmem_.w[0; }", 8, false));
    puts("heap-cache rewrite: read/store, audit, readonly, atomic and malformed fallback passed");
    if (argc == 3) {
        FILE *f = fopen(argv[1], "rb"); assert(f);
        fseek(f, 0, SEEK_END); long n = ftell(f); rewind(f);
        char *in = malloc((size_t)n+1); assert(in && fread(in, 1, n, f) == (size_t)n); fclose(f); in[n] = 0;
        s = shader_heap_cache(in, 53248, true); assert(s);
        f = fopen(argv[2], "wb"); assert(f); fputs(s, f); fclose(f); free(s); free(in);
    }
    return 0;
}
