// NVN implemented on the host. Guest NVN objects hold a pointer to a host object at offset 0;
// builders and small state objects store their fields directly in guest memory.
#pragma once
#include "../hle.h"
#include "tegra.h"

#define NVN(name) HLE_FUNC(#name, hle_##name)

// argument helpers
#define A(i) (c->x[(i)])
#define AP(i) ((void *)(uintptr_t)c->x[(i)])
#define AF(i) (*(float *)&c->v[(i)])
#define RET(v) (c->x[0] = (uint64_t)(v))
#define RETF(f) do { c->v[0] = 0; *(float *)&c->v[0] = (f); } while (0)
#define HOST(T, guest) ((T *)(uintptr_t)(*(uint64_t *)(uintptr_t)(guest)))
#define SET_HOST(guest, p) (*(uint64_t *)(uintptr_t)(guest) = (uint64_t)(uintptr_t)(p))
#define G(T, base, off) (*(T *)((uint8_t *)(uintptr_t)(base) + (off)))

typedef struct NvnPool {
    uint64_t storage, size;
    uint32_t flags;
} NvnPool;

typedef struct NvnTexture {
    TexLayout layout;
    uint32_t target, format, flags, width, height, depth, levels, samples, ds_mode;
    uint8_t swizzle[4];
    uint64_t stride;
    uint64_t addr;  // guest address of texel storage
    NvnPool *pool;
    uint64_t pool_offset;
    void *backend;  // renderer-side texture
    int backend_slot;  // (Vulkan backend) slot in textures.c's list of textures holding an image, + 1
    uint64_t backend_used_frame; // last use of this texture holder, not another alias of its image
    uint64_t debug_id;
    bool finalized;  // nvnTextureFinalize ran (the object stays: pool entries and copies may still name it)
    bool zombie;     // (Vulkan backend) in textures.c's list of finalized textures holding an image
    uint32_t init_gen;  // initializes of this host object (texture interning hands it back): a finalize queued before the last is dropped
    uint32_t cpu_levels;  // levels nvnTextureWriteTexels filled since the renderer last uploaded them (bit per level)
    char label[64];
} NvnTexture;
NvnTexture *nvn_texture_host(uint64_t guest);  // (objects.c) the host texture a guest NVNtexture holds now, or NULL

// Pitch (row-major, the builder's stride) storage instead of block-linear: NVN_TEXTURE_FLAGS_LINEAR (0x10) and
// NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET (0x40). BOTW's terrain height tiles are 256x256 R16 linear render targets
// the CPU fills row by row.
static inline bool nvn_texture_flags_pitch(uint32_t flags) { return (flags & (0x10 | 0x40)) != 0; }

typedef struct {
    uint32_t min_filter, mag_filter, wrap[3], compare_mode, compare_func, reduction;
    float lod_min, lod_max, lod_bias, max_aniso;
    float border[4];
    void *backend;
    float backend_aniso;  // the max anisotropy `backend` was made with (vulkan/textures.c vk_sampler)
    void *backend_plain;  // the same without Video > Anisotropic filtering's override (vk_sampler aniso_ok false)
    float backend_plain_aniso;
    float backend_bias;   // the LOD bias `backend` was made with on top of the game's (vk_sampler: DLSS rendering smaller)
} NvnSamplerDesc;

// Our layout inside the guest NVNtextureView. NVN 55's struct is 16 bytes (its driver's SetDefaults clears exactly
// that; BOTW 1.8.2 keeps views at the end of heap blocks), so ours must fit in it: NVN 53's 0x28 bytes overran the
// next block's heap header.
typedef struct {
    uint16_t min_layer, num_layers, format;
    uint8_t setmask;  // which fields are overridden: 1 levels, 2 layers, 4 format, 8 swizzle, 16 ds mode, 32 target
    uint8_t target, ds_mode, base_level, num_levels;
    uint8_t swizzle[4];
} NvnTextureView;
_Static_assert(sizeof(NvnTextureView) <= 16, "NVNtextureView is 16 bytes in NVN 55");

typedef struct {
    NvnTexture *tex;
    NvnTextureView view;
    bool has_view;
} NvnTexEntry;

typedef struct NvnDescPool {
    uint64_t addr;
    uint64_t revision;  // native descriptor caches invalidate on pool registration
    int count;
    void *entries;  // NvnTexEntry[] or NvnSamplerDesc[]
} NvnDescPool;

typedef struct NvnShaderStage {
    uint32_t stage;           // NVNshaderStage
    uint64_t code_addr;       // GPU (== CPU) address of the code
    const uint8_t *control;   // control section
    uint32_t const_offset, const_size;  // compiler constants (bound as c1), within the code blob
    uint64_t const_id;        // Vulkan immutable-constant identity; reset by ProgramSetShaders
    void *backend;
} NvnShaderStage;

typedef struct NvnProgram {
    NvnShaderStage stages[6];
    uint32_t stage_mask;
    void *backend;
} NvnProgram;

typedef struct NvnWindow {
    uint64_t textures[8];  // guest NVNtexture*
    int num, active, interval, next;
    int crop[4];
} NvnWindow;

typedef struct NvnSync {
    volatile int signaled;
    volatile uint64_t seq;  // GPU submission this fence belongs to (renderer_fence_seq), 0 = none
    uint64_t ticket;        // (objects.c submission queue) the queued nvnQueueFenceSync that sets seq, 0 = none
    volatile int in_list;   // a command list fences it (nvnCommandBufferFenceSync, set when it is recorded)
} NvnSync;

// ---- command recording (runtime/nvn/cmdbuf.c: records in host memory, accounted to guest command memory)
typedef struct {
    uint64_t gmem, gsize, gused;  // the guest command memory chunk, and the GPU bytes the game is told are used
    struct Shadow *shadow;        // host storage for what is recorded while that chunk is current
    int block;                    // the shadow's block in use
    uint64_t mem;    // that block (host address)
    uint64_t size;   // its size
    uint64_t used;   // bytes of records in it
    uint64_t start;  // first record of the list being recorded
    uint64_t callback, callback_data;
    bool recording;
    uint64_t guest;  // the guest NVNcommandBuffer*
} NvnCmdBuf;
void cmdbuf_pool_gone(uint64_t lo, uint64_t hi);  // a memory pool is finalized: lists recorded in it die

#include "command_record.h"



// renderer backend (runtime/nvn/metal.m)
void backend_init(void);
void backend_execute(uint64_t handle);            // run a recorded command list
void backend_present(NvnTexture *tex);            // show a texture in the window
void backend_texture_destroyed(NvnTexture *tex);
void backend_pool_written(uint64_t addr, uint64_t size);
void backend_debug_copies_at(uint64_t addr);  // (debugging) log the recent texture-to-buffer copies that wrote addr
void nvn_debug_textures_at(uint64_t addr);    // (objects.c, debugging) log the textures whose storage holds addr
