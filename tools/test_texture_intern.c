#include <assert.h>
#include <stdio.h>
#include "../runtime/nvn/texture_intern.h"

typedef struct { uint32_t w, h; } Shape;
static bool same(const void *host, const void *ctx) {
    const Shape *a = host, *b = ctx;
    return a->w == b->w && a->h == b->h;
}

int main(void) {
    TextureIntern m = {0};
    assert(!texture_intern_entry(&m, 0, true));
    assert(!texture_intern_entry(&m, 0x80001000, false));

    // Many guest objects: each keeps its own entry through growth.
    const uint32_t n = 200000;
    for (uint32_t i = 1; i <= n; i++) {
        TextureInternEntry *e = texture_intern_entry(&m, 0x80000000ull + (uint64_t)i * 0x40, true);
        assert(e && e->guest == 0x80000000ull + (uint64_t)i * 0x40 && e->n == 0);
        texture_intern_add(e, (void *)(uintptr_t)(i * 16));
    }
    assert(m.count == n);
    for (uint32_t i = 1; i <= n; i++) {
        TextureInternEntry *e = texture_intern_entry(&m, 0x80000000ull + (uint64_t)i * 0x40, false);
        assert(e && e->n == 1 && e->host[0] == (void *)(uintptr_t)(i * 16));
        assert(texture_intern_entry(&m, 0x80000000ull + (uint64_t)i * 0x40, true) == e);  // no duplicate
        assert(!texture_intern_entry(&m, 0x80000000ull + (uint64_t)i * 0x40 + 8, false));
    }
    assert(m.count == n);
    texture_intern_clear(&m);
    assert(!m.capacity && !texture_intern_entry(&m, 0x80000040, false));

    // One guest object, a chain of shapes initialized again and again (the exposure chain): each comes back.
    Shape chain[7];
    TextureInternEntry *e = texture_intern_entry(&m, 0x81230000, true);
    for (int i = 0; i < 7; i++) {
        chain[i] = (Shape){192u >> i, 192u >> i};
        assert(!texture_intern_find(e, same, &chain[i]));
        texture_intern_add(e, &chain[i]);
    }
    for (int frame = 0; frame < 3; frame++)
        for (int i = 0; i < 7; i++) {
            Shape want = {192u >> i, 192u >> i};
            assert(texture_intern_find(e, same, &want) == &chain[i]);
        }
    Shape other = {5, 7};
    assert(!texture_intern_find(e, same, &other));
    assert(!texture_intern_find(NULL, same, &other));

    // Past the ways, the oldest is forgotten first.
    Shape many[TEXTURE_INTERN_WAYS + 3];
    TextureInternEntry *f = texture_intern_entry(&m, 0x81240000, true);
    for (int i = 0; i < TEXTURE_INTERN_WAYS + 3; i++) {
        many[i] = (Shape){1000u + i, 1};
        texture_intern_add(f, &many[i]);
    }
    assert(f->n == TEXTURE_INTERN_WAYS);
    for (int i = 0; i < 3; i++) assert(!texture_intern_find(f, same, &many[i]));
    for (int i = 3; i < TEXTURE_INTERN_WAYS + 3; i++) assert(texture_intern_find(f, same, &many[i]) == &many[i]);
    texture_intern_add(NULL, &other);  // (no entry: nothing happens)
    texture_intern_add(f, NULL);
    assert(f->n == TEXTURE_INTERN_WAYS);
    texture_intern_clear(&m);
    puts("texture interning: 200000 guest objects, chain reuse, oldest-first replacement passed");
    return 0;
}
