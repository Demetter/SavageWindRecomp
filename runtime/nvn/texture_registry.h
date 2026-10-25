#pragma once
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// Host texture addresses stay valid until reboot. Independent shards keep a
// growth operation from rebuilding every texture created during the whole boot.
// The caller supplies synchronization, as with the previous single table.
#define TEXTURE_REGISTRY_SHARDS 256
typedef struct { uintptr_t *slots; uint32_t capacity, count; } TextureRegistryShard;
typedef struct { TextureRegistryShard shard[TEXTURE_REGISTRY_SHARDS]; } TextureRegistry;

static uint64_t texture_registry_hash(uintptr_t p) {
    uint64_t h = (uint64_t)p >> 4;
    h = (h ^ (h >> 30)) * 0xbf58476d1ce4e5b9ull;
    h = (h ^ (h >> 27)) * 0x94d049bb133111ebull;
    return h ^ (h >> 31);
}

static bool texture_registry_insert(TextureRegistry *r, uintptr_t p, uint32_t *moved) {
    if (moved) *moved = 0;
    if (!p) return false;
    uint64_t hash = texture_registry_hash(p);
    TextureRegistryShard *s = &r->shard[hash & (TEXTURE_REGISTRY_SHARDS - 1)];
    if ((uint64_t)(s->count + 1) * 2 > s->capacity) {
        if (s->capacity > UINT32_MAX / 2) return false;
        uint32_t cap = s->capacity ? s->capacity * 2 : 64;
        uintptr_t *next = calloc(cap, sizeof *next);
        if (!next) return false;
        for (uint32_t i = 0; i < s->capacity; i++) if (s->slots[i]) {
            uintptr_t value = s->slots[i];
            uint32_t slot = (uint32_t)(texture_registry_hash(value) >> 8) & (cap - 1);
            while (next[slot]) slot = (slot + 1) & (cap - 1);
            next[slot] = value;
        }
        if (moved) *moved = s->count;
        free(s->slots);
        s->slots = next; s->capacity = cap;
    }
    uint32_t i = (uint32_t)(hash >> 8) & (s->capacity - 1);
    while (s->slots[i]) {
        if (s->slots[i] == p) return true;
        i = (i + 1) & (s->capacity - 1);
    }
    s->slots[i] = p; s->count++;
    return true;
}

static bool texture_registry_contains(const TextureRegistry *r, uintptr_t p) {
    if (!p) return false;
    uint64_t hash = texture_registry_hash(p);
    const TextureRegistryShard *s = &r->shard[hash & (TEXTURE_REGISTRY_SHARDS - 1)];
    if (!s->capacity) return false;
    for (uint32_t i = (uint32_t)(hash >> 8) & (s->capacity - 1); s->slots[i]; i = (i + 1) & (s->capacity - 1))
        if (s->slots[i] == p) return true;
    return false;
}

static void texture_registry_clear(TextureRegistry *r, void (*destroy)(void *)) {
    for (unsigned n = 0; n < TEXTURE_REGISTRY_SHARDS; n++) {
        TextureRegistryShard *s = &r->shard[n];
        if (destroy) for (uint32_t i = 0; i < s->capacity; i++) if (s->slots[i]) destroy((void *)s->slots[i]);
        free(s->slots);
    }
    memset(r, 0, sizeof *r);
}
