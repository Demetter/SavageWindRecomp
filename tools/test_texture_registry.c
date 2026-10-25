#include <assert.h>
#include <stdio.h>
#include "../runtime/nvn/texture_registry.h"

static size_t destroyed;
static void count_destroy(void *p) { assert(p); destroyed++; }
int main(void) {
    TextureRegistry r = {0};
    const uint32_t n = 1000000;
    uint32_t largest_move = 0, moved;
    assert(!texture_registry_contains(&r, 0));
    assert(!texture_registry_insert(&r, 0, &moved));
    for (uint32_t i = 1; i <= n; i++) {
        uintptr_t p = 0x100000000ull + (uintptr_t)i * 256;
        assert(texture_registry_insert(&r, p, &moved));
        if (moved > largest_move) largest_move = moved;
        assert(texture_registry_contains(&r, p));
    }
    for (uint32_t i = 1; i <= n; i++) {
        uintptr_t p = 0x100000000ull + (uintptr_t)i * 256;
        assert(texture_registry_contains(&r, p));
        assert(!texture_registry_contains(&r, p + 16));
        if (!(i % 1000)) assert(texture_registry_insert(&r, p, NULL));
    }
    size_t total = 0;
    for (unsigned s = 0; s < TEXTURE_REGISTRY_SHARDS; s++) total += r.shard[s].count;
    assert(total == n && largest_move <= 8192);
    texture_registry_clear(&r, count_destroy);
    assert(destroyed == n && !texture_registry_contains(&r, 0x100000100ull));
    assert(texture_registry_insert(&r, 0x100000100ull, NULL));
    assert(texture_registry_contains(&r, 0x100000100ull));
    texture_registry_clear(&r, NULL);
    printf("texture registry: one million entries, duplicates, misses and reboot passed; largest growth moved %u entries\n", largest_move);
}
