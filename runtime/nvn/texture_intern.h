#pragma once
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// The host textures already made for each guest NVNtexture object (objects.c, SAVAGEWIND_TEXTURE_INTERN). BOTW
// initializes the same objects again every frame with the same few shapes, and each initialize made a host object that
// lives until reboot. Up to TEXTURE_INTERN_WAYS shapes are remembered per guest object; past that the oldest is
// forgotten (its host object stays valid, as every host texture does). The caller supplies synchronization and the
// shape comparison.
#define TEXTURE_INTERN_WAYS 16
typedef struct { uint64_t guest; uint32_t n, next; void *host[TEXTURE_INTERN_WAYS]; } TextureInternEntry;
typedef struct { TextureInternEntry **slots; uint32_t capacity, count; } TextureIntern;

static uint64_t texture_intern_hash(uint64_t guest) {
    uint64_t h = guest >> 3;
    h = (h ^ (h >> 30)) * 0xbf58476d1ce4e5b9ull;
    h = (h ^ (h >> 27)) * 0x94d049bb133111ebull;
    return h ^ (h >> 31);
}

// guest's entry; with create, a new empty one when it has none. NULL: none (or out of memory).
static TextureInternEntry *texture_intern_entry(TextureIntern *m, uint64_t guest, bool create) {
    if (!guest) return NULL;
    if (m->capacity)
        for (uint32_t i = (uint32_t)texture_intern_hash(guest) & (m->capacity - 1); m->slots[i]; i = (i + 1) & (m->capacity - 1))
            if (m->slots[i]->guest == guest) return m->slots[i];
    if (!create) return NULL;
    if ((uint64_t)(m->count + 1) * 2 > m->capacity) {
        if (m->capacity > UINT32_MAX / 2) return NULL;
        uint32_t cap = m->capacity ? m->capacity * 2 : 256;
        TextureInternEntry **next = calloc(cap, sizeof *next);
        if (!next) return NULL;
        for (uint32_t i = 0; i < m->capacity; i++) if (m->slots[i]) {
            uint32_t s = (uint32_t)texture_intern_hash(m->slots[i]->guest) & (cap - 1);
            while (next[s]) s = (s + 1) & (cap - 1);
            next[s] = m->slots[i];
        }
        free(m->slots);
        m->slots = next; m->capacity = cap;
    }
    TextureInternEntry *e = calloc(1, sizeof *e);
    if (!e) return NULL;
    e->guest = guest;
    uint32_t i = (uint32_t)texture_intern_hash(guest) & (m->capacity - 1);
    while (m->slots[i]) i = (i + 1) & (m->capacity - 1);
    m->slots[i] = e; m->count++;
    return e;
}

// The first host object of e that same(host, ctx) accepts, or NULL.
static void *texture_intern_find(const TextureInternEntry *e, bool (*same)(const void *host, const void *ctx), const void *ctx) {
    for (uint32_t i = 0; e && i < e->n; i++)
        if (same(e->host[i], ctx)) return e->host[i];
    return NULL;
}

// Remembers host for e's guest object, in place of the oldest once all ways are taken.
static void texture_intern_add(TextureInternEntry *e, void *host) {
    if (!e || !host) return;
    if (e->n < TEXTURE_INTERN_WAYS) { e->host[e->n++] = host; return; }
    e->host[e->next] = host;
    e->next = (e->next + 1) % TEXTURE_INTERN_WAYS;
}

// Forgets every entry (the host objects belong to the caller).
static void texture_intern_clear(TextureIntern *m) {
    for (uint32_t i = 0; i < m->capacity; i++) free(m->slots[i]);
    free(m->slots);
    memset(m, 0, sizeof *m);
}
