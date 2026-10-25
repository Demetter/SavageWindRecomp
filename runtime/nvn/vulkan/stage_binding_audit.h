#pragma once
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <vulkan/vulkan.h>

#define STAGE_AUDIT_MAX_BIND 96
typedef struct { uint32_t n; uint64_t binding[STAGE_AUDIT_MAX_BIND][4]; } StageBindingAudit;
typedef const void *(*StageAuditUpload)(uint64_t buffer, uint64_t offset, uint64_t size);

// A memo collision may allocate an identical immutable uniform at a new address.
// Accept this only after comparing its actual bytes. Images and GPU-written
// storage must retain exact identities; the resolver only exposes live uploads.
static bool stage_audit_equal(const StageBindingAudit *a, const uint32_t *ad, int an,
                              const StageBindingAudit *b, const uint32_t *bd, int bn,
                              StageAuditUpload upload) {
    if (!a || !b || a->n != b->n || a->n > STAGE_AUDIT_MAX_BIND || an != bn || an < 0 || an > 32) return false;
    int dynamic = 0;
    for (uint32_t i = 0; i < a->n; i++) {
        const uint64_t *x = a->binding[i], *y = b->binding[i];
        if (x[0] != y[0]) return false;
        uint32_t type = (uint32_t)(x[0] >> 32);
        if (type != VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER && type != VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC) {
            if (memcmp(x, y, sizeof a->binding[i])) return false;
            continue;
        }
        if (x[3] != y[3]) return false;
        uint64_t xo = x[2], yo = y[2];
        if (type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC) {
            int k = 0;
            for (uint32_t j = 0; j < a->n; j++)
                if ((uint32_t)(a->binding[j][0] >> 32) == type && (uint32_t)a->binding[j][0] < (uint32_t)x[0]) k++;
            if (k >= an || UINT64_MAX - xo < ad[k] || UINT64_MAX - yo < bd[k]) return false;
            xo += ad[k]; yo += bd[k]; dynamic++;
        }
        if (x[1] == y[1] && xo == yo) continue;
        const void *xp = upload(x[1], xo, x[3]), *yp = upload(y[1], yo, y[3]);
        if (!xp || !yp || memcmp(xp, yp, (size_t)x[3])) return false;
    }
    return dynamic == an;
}
