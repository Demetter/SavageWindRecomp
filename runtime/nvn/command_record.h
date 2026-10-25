#pragma once
#include <stddef.h>
#include <stdint.h>
#include <string.h>

// Host-only records. Integer arguments retain their offsets; compact state
// commands stop at their last consumed argument, then carry the copied payload.
// Draw/float/sync/copy commands keep the complete prefix. Guest command-memory
// accounting and the lifetime of recorded payloads are independent of this form.
typedef struct {
    uint16_t op, header_size;
    uint32_t size;
    uint64_t x[7];
    float f[4];
    uint8_t extra[];
} CmdRec;
_Static_assert(sizeof(CmdRec) == 80 && offsetof(CmdRec, x) == 8 && offsetof(CmdRec, f) == 64,
               "command argument offsets must stay stable");

enum {
    CMD_END = 0, CMD_JUMP, CMD_CALL,
    CMD_BIND_PROGRAM, CMD_BIND_BLEND, CMD_BIND_CHANNEL_MASK, CMD_BIND_COLOR, CMD_BIND_DEPTH_STENCIL,
    CMD_BIND_MULTISAMPLE, CMD_BIND_POLYGON, CMD_BIND_VERTEX_ATTRIB, CMD_BIND_VERTEX_STREAM,
    CMD_BIND_VERTEX_BUFFERS, CMD_BIND_UNIFORM_BUFFERS, CMD_BIND_STORAGE_BUFFERS, CMD_BIND_TEXTURES,
    CMD_BIND_IMAGES, CMD_SET_TEXTURE_POOL, CMD_SET_SAMPLER_POOL, CMD_SET_RENDER_TARGETS,
    CMD_SET_VIEWPORTS, CMD_SET_SCISSORS, CMD_SET_DEPTH_RANGES, CMD_SET_BLEND_COLOR, CMD_SET_STENCIL,
    CMD_SET_PRIMITIVE_RESTART, CMD_SET_ALPHA_REF, CMD_SET_DEPTH_BOUNDS, CMD_SET_POLYGON_OFFSET,
    CMD_SET_LINE_WIDTH, CMD_SET_POINT_SIZE, CMD_SET_SAMPLE_MASK, CMD_SET_RENDER_ENABLE,
    CMD_CLEAR_COLOR, CMD_CLEAR_DEPTH_STENCIL, CMD_CLEAR_TEXTURE, CMD_CLEAR_BUFFER,
    CMD_DRAW_ARRAYS, CMD_DRAW_ELEMENTS, CMD_DRAW_ARRAYS_INDIRECT, CMD_DRAW_ELEMENTS_INDIRECT,
    CMD_DISPATCH, CMD_DISPATCH_INDIRECT,
    CMD_COPY_BUFFER_TO_BUFFER, CMD_COPY_BUFFER_TO_TEXTURE, CMD_COPY_TEXTURE_TO_BUFFER,
    CMD_COPY_TEXTURE_TO_TEXTURE, CMD_DOWNSAMPLE, CMD_UPDATE_UNIFORM_BUFFER, CMD_FENCE_SYNC, CMD_WAIT_SYNC,
    CMD_BARRIER, CMD_SIGNAL_EVENT, CMD_WAIT_EVENT, CMD_REPORT_COUNTER, CMD_DISCARD_COLOR,
    CMD_DISCARD_DEPTH_STENCIL, CMD_SET_COPY_STRIDES, CMD_DRAW_TEXTURE, CMD_RESOLVE_DEPTH, CMD_BIND_TEXTURES_DEFERRED,
    CMD_SET_DEPTH_CLAMP, CMD_RESET_COUNTER, CMD_SET_RASTERIZER_DISCARD,
    CMD_COUNT
};

static inline uint16_t nvn_cmd_compact_header(uint32_t op) {
    switch (op) {
    case CMD_END:
    case CMD_BIND_BLEND: case CMD_BIND_CHANNEL_MASK: case CMD_BIND_COLOR:
    case CMD_BIND_DEPTH_STENCIL: case CMD_BIND_MULTISAMPLE: case CMD_BIND_POLYGON:
    case CMD_SET_BLEND_COLOR:
        return 8;
    case CMD_CALL: case CMD_BIND_PROGRAM: case CMD_BIND_VERTEX_ATTRIB: case CMD_BIND_VERTEX_STREAM:
    case CMD_SET_TEXTURE_POOL: case CMD_SET_SAMPLER_POOL: case CMD_SET_RENDER_TARGETS:
    case CMD_BARRIER: case CMD_SET_DEPTH_CLAMP: case CMD_SET_RENDER_ENABLE: case CMD_SET_RASTERIZER_DISCARD:
    case CMD_RESET_COUNTER:
        return 16;
    case CMD_BIND_VERTEX_BUFFERS: case CMD_SET_VIEWPORTS: case CMD_SET_SCISSORS: case CMD_SET_DEPTH_RANGES:
    case CMD_SET_PRIMITIVE_RESTART: case CMD_SET_COPY_STRIDES:
        return 24;
    case CMD_BIND_UNIFORM_BUFFERS: case CMD_BIND_STORAGE_BUFFERS: case CMD_BIND_TEXTURES: case CMD_BIND_IMAGES:
        return 32;
    case CMD_BIND_TEXTURES_DEFERRED:
        return 40;
    default:
        return sizeof(CmdRec);
    }
}

static inline uint8_t *nvn_cmd_extra(const CmdRec *r) {
    return (uint8_t *)(uintptr_t)r + (r->header_size ? r->header_size : sizeof *r);
}

static inline void nvn_cmd_init(CmdRec *r, uint32_t op, uint32_t size, uint16_t header) {
    memset(r, 0, header);
    r->op = (uint16_t)op;
    r->header_size = header;
    r->size = size;
}

static inline void nvn_cmd_arguments(CmdRec *r, const uint64_t x[7], const float f[4]) {
    if (r->header_size == sizeof *r) {
        memcpy(r->x, x, sizeof r->x);
        memcpy(r->f, f, sizeof r->f);
    } else if (r->header_size > 8) {
        memcpy(r->x, x, r->header_size - 8);
    }
}

// Debug logs must not read a discarded argument or the following compact record.
static inline uint64_t nvn_cmd_debug_arg(const CmdRec *r, unsigned i) {
    unsigned header = r->header_size ? r->header_size : sizeof *r;
    return i < 7 && 8 + (i + 1) * 8 <= header ? r->x[i] : 0;
}
