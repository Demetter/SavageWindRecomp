#pragma once
#include <stdint.h>

// Experimental building block, not enabled in gameplay. Each draw retains its
// own addresses and byte limits; this is not a cache of mutable guest contents.
typedef struct {
    uint64_t address;
    uint32_t words;
    uint32_t reserved;
} NativeDrawConstant;
typedef struct { NativeDrawConstant constants[36]; } NativeDrawConstants;

// Rewrite only the translator's recognized CB declarations/scalar loads. The
// caller supplies a trusted draw-index expression and must provide set 2 binding
// 0 with NativeDrawConstants records. Unknown syntax fails closed (NULL).
// stage: 0 vertex, 1 fragment. Caller owns result. Not a full shader variant:
// draw ID propagation, buffer lifetime/dependencies and layouts remain its job.
char *vk_shader_native_draw_constants(const char *source, unsigned stage,
                                     const char *draw_index, uint32_t *mask);

// Graphics variant: propagate gl_DrawID + a batch base as a flat uint from VS
// to FS. The caller chooses a location unused by BOTH stages and enables Vulkan
// shaderDrawParameters. No vertex/index builtin is changed by this function.
// Check device interface-component limits INCLUDING builtins; a free location
// alone does not establish that the extra varying fits (tested on RTX5090).
// The vertex layout needs a four-byte vertex-stage push constant at offset 0.
char *vk_shader_native_draw_graphics(const char *source, unsigned stage,
                                    unsigned location, uint32_t *mask);
