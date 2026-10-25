#include "../runtime/nvn/vulkan/shader_constants.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    const uint32_t bits[] = {0x80000000u, 0x7fc12345u, 0x3e800000u, 0xffffffffu, 0x12345678u};
    const char *decl = "layout(std140, set = 1, binding = 1) uniform CB1 { uvec4 cb1[16]; };\n";
    char input[1024];
    snprintf(input, sizeof input, "%suint f(){return cb1[0][0]^cb1[0][1]^cb1[1][0]^cb10[0][0];}\n", decl);
    char *out = vk_shader_literal_constants(input, (const uint8_t *)bits, sizeof bits);
    assert(out && !strstr(out, "uniform CB1") && strstr(out, "0x80000000u") && strstr(out, "0x7fc12345u") &&
           strstr(out, "0x12345678u") && strstr(out, "cb10[0][0]"));
    free(out);
    const char *bad[] = {"cbr(cb1, r0)", "cb1[1][1]", "cb1[999999999999999999999999][0]", "cb1[r0][0]", "cb1[0]", "cb1[0][4]", "cb1"};
    for (unsigned i = 0; i < sizeof bad / sizeof bad[0]; i++) {
        snprintf(input, sizeof input, "%suint f(){return %s;}\n", decl, bad[i]);
        assert(!vk_shader_literal_constants(input, (const uint8_t *)bits, sizeof bits));
    }
    snprintf(input, sizeof input, "%suint f(){return cb1[0][0];}\n", decl);
    out = vk_shader_literal_constants(input, (const uint8_t *)bits, 4);
    assert(out && strstr(out, "0x80000000u")); free(out);
    assert(!vk_shader_literal_constants(input, (const uint8_t *)bits, 3));
    assert(!vk_shader_literal_constants("uint cb10;", (const uint8_t *)bits, sizeof bits));
    puts("Static compiler constants: exact bits, declared bounds and fallback cases passed.");
    return 0;
}
