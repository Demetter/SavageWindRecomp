// Actual translation-cache serialization and literal specialization under both
// buffer-range policies. No GPU or player's cache is used.
#include <assert.h>
#include <direct.h>
#include "../runtime/nvn/vulkan/shaders.c"

void rt_fatal_trace(void) { abort(); }
// Unused renderer/platform paths in the included production translation unit.
// Fail if this cache-only test accidentally reaches them.
VkG g_vk;
const char smaa_hlsl_src[] = "";
const char *vk_spirv_id(void) { return "test-spirv"; }
bool plat_remove_tree(const char *p) { (void)p; abort(); }
bool vk_replace_file(const char *a, const char *b) { (void)a; (void)b; abort(); }
void vk_device_fault_report(void) { abort(); }
int vk_spirv_ensure(const char *s, int stage) { (void)s; (void)stage; abort(); }
uint32_t *vk_glsl_to_spirv(const char *s, int stage, const char *n, size_t *words) {
    (void)s; (void)stage; (void)n; (void)words; abort();
}
uint64_t shader_code_hash(const uint8_t *p, size_t n, int stage, uint32_t *size) {
    (void)p; (void)n; (void)stage; (void)size; abort();
}
int main(void) {
    _putenv_s("SAVAGEWIND_CACHE_DIR", "build/codex-perf/xlate-policy-test");
    _putenv_s("SAVAGEWIND_NO_XLATE_CACHE", "");
    _putenv_s("SAVAGEWIND_VK_SHADER_CONSTANTS", "1");
    const uint64_t code[8] = {0, 0xe30000000007000full, 0, 0, 0, 0, 0, 0};
    const uint32_t constants[4] = {0x80000000u, 0x7fc12345u, 0xffffffffu, 42};
    ShaderKey key = {.glsl = true};
    uint64_t ids[2], derived[2];
    XlateHeader hd[2];
    const char *source = "layout(std140, set = 0, binding = 1) uniform CB1 { uvec4 cb1[1]; };\n"
                         "uint f(){return cb1[0][0]^cb1[0][1]^cb1[0][2]^cb1[0][3];}\n";
    for (unsigned policy = 0; policy < 2; policy++) {
        _putenv_s("SAVAGEWIND_CBR", policy ? "1" : "0");
        assert(maxwell_translation_policy() == policy && !maxwell_translation_uncacheable() && !xlate_off());
        ids[policy] = xlate_id((const uint8_t *)code, sizeof code, STAGE_COMPUTE, &key, &hd[policy]);
        ShaderInfo *base = calloc(1, sizeof *base);
        base->stage = STAGE_COMPUTE; base->msl = strdup(source); base->cbuf_mask = 2; base->cbuf_size[1] = 16;
        hd[policy].msl_len = (uint32_t)strlen(source);
        xlate_store(ids[policy], &hd[policy], base);
        ShaderInfo *disk = xlate_load(ids[policy], NULL, STAGE_COMPUTE);
        assert(disk && !strcmp(disk->msl, source)); maxwell_free(disk);
        xlate_keep(ids[policy], base);
        ShaderInfo *specialized = vk_translate_program((const uint8_t *)code, sizeof code, STAGE_COMPUTE,
                                                       &key, (const uint8_t *)constants, sizeof constants, false);
        assert(specialized && !(specialized->cbuf_mask & 2) && !strstr(specialized->msl, "uniform CB1"));
        assert(strstr(specialized->msl, "0x80000000u") && strstr(specialized->msl, "0x7fc12345u"));
        derived[policy] = vk_translate_id(specialized);
        assert(derived[policy] && derived[policy] != ids[policy]);
        disk = xlate_load(derived[policy], &hd[policy], STAGE_COMPUTE);
        assert(disk && !strcmp(disk->msl, specialized->msl)); maxwell_free(disk);
    }
    assert(ids[0] != ids[1] && derived[0] != derived[1]);
    for (unsigned policy = 0; policy < 2; policy++) {
        _putenv_s("SAVAGEWIND_CBR", policy ? "1" : "0");
        assert(!xlate_load(ids[1-policy], NULL, STAGE_COMPUTE));
        assert(!xlate_load(derived[1-policy], NULL, STAGE_COMPUTE));
        assert(!xlate_load(ids[1-policy], &hd[policy], STAGE_COMPUTE));
    }
    _putenv_s("SAVAGEWIND_CBR", "");
    assert(maxwell_translation_policy() == 0);
    _putenv_s("SAVAGEWIND_CBR_DUMP", "diagnostic");
    assert(maxwell_translation_uncacheable());
    puts("Translation policies: separate keys, exact disk reload, cross-policy rejection and c1 specialization passed.");
    return 0;
}
