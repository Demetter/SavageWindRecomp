// Exercise the real compiler and disk cache with both policies. An optimized
// entry must never satisfy an unoptimized lookup of the same stage/source.
#include <assert.h>
#include <direct.h>
#include "../runtime/nvn/vulkan/spirv.c"

void rt_fatal_trace(void) { abort(); }
const char *vk_cache_dir_id(const char *id) { (void)id; return "policy-test"; }
void vk_cache_path(char *out, size_t n, const char *name) {
    snprintf(out, n, "build/codex-perf/spirv-policy-test/%s", name);
}

int main(void) {
    _mkdir("build/codex-perf/spirv-policy-test");
    const char *src = "#version 460\nlayout(local_size_x=1) in;\n"
        "layout(std430,set=0,binding=17) buffer B {uint v[];} b;\n"
        "void main(){uint x=b.v[0]; uint y=x+1u; if(x>2u)y+=3u; b.v[1]=y;}";
    uint64_t h[2] = {src_hash(src, STAGE_COMPUTE, false), src_hash(src, STAGE_COMPUTE, true)};
    assert(h[0] != h[1]);
    char paths[2][512];
    for (unsigned v = 0; v < 2; v++) spirv_path(paths[v], sizeof paths[v], h[v]);
    assert(strcmp(paths[0], paths[1]));
    size_t words[2];
    uint32_t *compiled[2];
    for (unsigned v = 0; v < 2; v++) {
        compiled[v] = vk_glsl_to_spirv_mode(src, STAGE_COMPUTE, "policy-test", &words[v], v != 0);
        assert(compiled[v] && compiled[v][0] == 0x07230203);
        size_t n = 0;
        uint32_t *disk = cache_load(h[v], &n);
        assert(disk && n == words[v] && !memcmp(disk, compiled[v], n * 4));
        free(disk);
    }
    // This shader has compiler-eliminated local loads/stores, ensuring the test
    // actually produced distinct compiler products rather than two identical files.
    assert(words[0] != words[1] || memcmp(compiled[0], compiled[1], words[0] * 4));
    for (unsigned v = 0; v < 2; v++) {
        size_t n = 0;
        uint32_t *again = vk_glsl_to_spirv_mode(src, STAGE_COMPUTE, "policy-test", &n, v != 0);
        assert(again && n == words[v] && !memcmp(again, compiled[v], n * 4));
        free(again); free(compiled[v]);
    }
    puts("SPIR-V policies: distinct compiler output, separate disk keys, exact warm-cache reloads passed");
    return 0;
}
