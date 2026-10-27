// Sharper bitmap text at high output resolutions.
//
// BOTW sets its text with bitmap fonts (Font/Font_<region>.sbfarc): Normal_00 for dialogue and most menus (31x39
// cells), NormalS_00 for small print (A8), Caption_00, Special_00 for titles, Ancient_00 and External_00 (the button
// icons: runtime/nvn/glyphs.c). Their sheets, mostly 1024x1024 BC4, are made for a 1280x720 layout drawn at 1920x1080:
// at 2160 lines each texel covers three pixels, and the bilinear filter shows it as soft edges.
//
// SAVAGEWIND_FONT_SCALE=N (2..4; auto, the default: the text's size on screen, made again when the output size changes
// it; 1: off) makes those
// images N times as wide and as high (textures.c) and fills them from the game's own glyphs: the signed distance to
// each glyph's edge, worked out from its anti-aliased coverage (the edge estimate of Gustavson & Strand's anti-aliased
// Euclidean distance transform, 2011), is interpolated to the larger grid and made coverage again with an edge 1.5 new
// texels wide, moved out by 0.2 old texels to keep the strokes' weight (font_edge, font_bold). Only the edges get
// finer; square corners round off by about a third of an original texel. Features too thin for an edge of their own
// (no texel of theirs nearly covered) keep their coverage, interpolated. NormalS_00's outlined glyphs are two layers,
// each with its edges. The game samples the sheets with normalized coordinates, so nothing else changes (texel
// fetches and size queries go through the texture's scale, as for scaled render targets).
//
// A sheet is made on the cores when it first goes up (10-20 ms for 1024x1024 at 4x on 16 threads; a texture holds up
// to 9 of them: Special_00 in Font_EU). N well above the text's size on screen (4 at 1080 lines) makes the edges
// thinner than a pixel and curves a little jagged (the sheets have no mipmaps): about window height / 720, which is
// what auto picks (2 for 1080 or 1440 lines, 3 for 2160).
//
// Nothing of the fonts is shipped: the sheets are recognised by the hashes of their texels (k_sheets;
// tools/font_sheet_hashes.py prints them from a RomFS folder) and made from the player's own when they go up.
#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#include <SDL3/SDL.h>

#include "font_scale.h"
#include "../rt.h"

#define FMT_BC4 0x49
#define FMT_R8 0x01

int font_scale_setting(void) {
    static int n = -1;
    if (n < 0) {
        const char *e = getenv("SAVAGEWIND_FONT_SCALE");  // (default auto: from the output resolution)
        n = !e || !*e || !strcmp(e, "auto") ? 0 : atoi(e);
        if (n < 1 || n > 4) n = 0;
    }
    return n;
}

// ---- recognising the sheets
// NVN format, size and hash (sheet_hash) of each sheet's texels as stored: tools/font_sheet_hashes.py on the RomFS of
// 1.9.0 (1.8.2's fonts are the same). One row per distinct sheet: Ancient_00 and External_00 are the same everywhere.
typedef struct { uint16_t format, w, h; uint64_t hash; } Sheet;
static const Sheet k_sheets[] = {
    {0x49, 32, 1024, 0x67598bdd808d3f93ull},    // Ancient_00 0 (every region)
    {0x49, 256, 1024, 0xe100d4d52110760bull},   // External_00 0 (every region)
    {0x49, 1024, 1024, 0xcb326012977f67f5ull},  // Font_EU Special_00 0
    {0x49, 1024, 1024, 0x03ead4561a5c1a93ull},  // Font_EU Special_00 1
    {0x49, 1024, 1024, 0x13220aca38a9ddecull},  // Font_EU Special_00 2
    {0x49, 1024, 1024, 0xac5ac917fc733a7full},  // Font_EU Special_00 3
    {0x49, 1024, 1024, 0xdc3e1bb2793a3885ull},  // Font_EU Special_00 4
    {0x49, 1024, 1024, 0x337a287e7e87d5bbull},  // Font_EU Special_00 5
    {0x49, 1024, 1024, 0x75e8c55e455e1c7bull},  // Font_EU Special_00 6
    {0x49, 1024, 1024, 0x1bf9e55a02b9efb7ull},  // Font_EU Special_00 7
    {0x49, 1024, 1024, 0x0f52ea496ea13f27ull},  // Font_EU Special_00 8
    {0x49, 1024, 1024, 0x2a42c2bf87bc36cfull},  // Font_EU Caption_00 0
    {0x49, 1024, 1024, 0x6e24b7a07f9faa64ull},  // Font_EU Normal_00 0
    {0x49, 1024, 1024, 0xded2553e503ac607ull},  // Font_EU Normal_00 1
    {0x49, 1024, 1024, 0x768c6bdcb2d0354dull},  // Font_EU Normal_00 2
    {0x49, 1024, 1024, 0xe874266cfdf71514ull},  // Font_EU Normal_00 3
    {0x01, 1024, 1024, 0x806396f5fde848c3ull},  // Font_EU NormalS_00 0
    {0x01, 1024, 1024, 0x1aa04ed1a1083cabull},  // Font_EU NormalS_00 1
    {0x49, 1024, 1024, 0x39255c3178d78a88ull},  // Font_JP Special_00 0
    {0x49, 1024, 1024, 0xf54992dd1c0502b6ull},  // Font_JP Special_00 1
    {0x49, 1024, 1024, 0xc2c7d4619444af24ull},  // Font_JP Special_00 2
    {0x49, 1024, 1024, 0x72a34badd0863762ull},  // Font_JP Special_00 3
    {0x49, 1024, 1024, 0xb7ac637364b7db8dull},  // Font_JP Special_00 4
    {0x49, 1024, 1024, 0x27ceeda8c5326ec7ull},  // Font_JP Special_00 5
    {0x49, 1024, 1024, 0x8191ba2907b372b1ull},  // Font_JP Caption_00 0
    {0x49, 1024, 1024, 0x8d71677085fd97c5ull},  // Font_JP Normal_00 0
    {0x49, 1024, 1024, 0x43f957d326adb7c8ull},  // Font_JP Normal_00 1
    {0x49, 1024, 1024, 0xceac4bda8d52b3cdull},  // Font_JP Normal_00 2
    {0x49, 1024, 1024, 0x3157e5d763db63d2ull},  // Font_JP Normal_00 3
    {0x01, 1024, 1024, 0xd07c08af9f348868ull},  // Font_JP NormalS_00 0
    {0x01, 1024, 1024, 0x7dafc5b4afd29105ull},  // Font_JP NormalS_00 1
    {0x49, 1024, 1024, 0xc219ca8739945940ull},  // Font_US Special_00 0
    {0x49, 1024, 1024, 0x872450a1613673b7ull},  // Font_US Special_00 1
    {0x49, 512, 1024, 0x53a5c01b4f68e0c6ull},   // Font_US Caption_00 0
    {0x49, 1024, 1024, 0x3a8cf4d57805aa58ull},  // Font_US Normal_00 0
    {0x49, 1024, 1024, 0xd7a96817a87cf820ull},  // Font_US Normal_00 1
    {0x01, 1024, 1024, 0x7e03c3251a09aec0ull},  // Font_US NormalS_00 0
};
#define NSHEETS (sizeof k_sheets / sizeof *k_sheets)

static uint64_t sheet_hash(const uint8_t *p, uint64_t n) {  // FNV-1a over 8-byte words (a byte at a time: 4x slower)
    uint64_t h = 0xcbf29ce484222325ull;
    for (uint64_t i = 0; i + 8 <= n; i += 8) {
        uint64_t w;
        memcpy(&w, p + i, 8);
        h = (h ^ w) * 0x100000001b3ull;
    }
    return h;
}

static bool sheet_known(uint32_t format, uint32_t w, uint32_t h, const uint64_t *hash) {
    for (size_t i = 0; i < NSHEETS; i++)
        if (k_sheets[i].format == format && k_sheets[i].w == w && k_sheets[i].h == h && (!hash || k_sheets[i].hash == *hash))
            return true;
    return false;
}

bool font_sheet_is(const uint8_t *mem, uint64_t layer_bytes, uint32_t layers, uint32_t format, uint32_t w, uint32_t h,
                   uint32_t levels) {
    if (levels > 1 || !layers || !layer_bytes || !sheet_known(format, w, h, NULL)) return false;
    for (uint32_t l = 0; l < layers; l++) {
        uint64_t hash = sheet_hash(mem + (uint64_t)l * layer_bytes, layer_bytes);
        if (!sheet_known(format, w, h, &hash)) {
            static int nlog;
            if (nlog++ < 8)
                LOG("[fonts] %ux%u texture (format %#x, layer %u of %u, %llu bytes) is no font sheet (hash %016llx)\n", w, h,
                    format, l, layers, (unsigned long long)layer_bytes, (unsigned long long)hash);
            return false;
        }
    }
    return true;
}

// ---- the larger sheet
// The sheet as layers of coverage, each with edges of its own: one for the BC4 sheets; two for the A8 ones
// (NormalS_00), whose glyphs are outlined: a band at half value around the full letter, both edges anti-aliased.
#define MAX_LAYERS 2
// A texel of a layer: the signed distance from its centre to the layer's edge, in texels (> 0 outside); how far to
// follow that edge (1: a stroke; 0: too thin for one, the coverage itself); its coverage
typedef struct { float s, k, a; } Tx;
typedef struct Up {
    uint32_t format, w, h, n, ow, oh, layers;  // the sheet's size, the factor, the new size
    const uint8_t *lin;
    uint8_t *out;
    float *a[MAX_LAYERS];       // coverage, 0..1
    float *across[MAX_LAYERS];  // the most coverage within 2 texels to either side
    Tx *t[MAX_LAYERS];
    uint8_t *flat[MAX_LAYERS];  // 0 / 2: the texel and its neighbours right, below, right below all empty / all covered
    int *x0, *x1, *y0, *y1;     // the texels each new column / row is interpolated from, and how far between them
    float *fx, *fy;
} Up;

static float clampf(float v, float lo, float hi) { return v < lo ? lo : v > hi ? hi : v; }

static void put(Up *u, uint64_t i, float v) {  // texel i's value (0..1) into its layers
    for (uint32_t l = 0; l < u->layers; l++) u->a[l][i] = clampf(v * u->layers - l, 0, 1);
}

static void decode_row(Up *u, uint32_t r) {  // (BC4: a row of blocks)
    uint32_t rows = 1;
    if (u->format == FMT_R8) {
        for (uint32_t x = 0; x < u->w; x++) put(u, (uint64_t)r * u->w + x, u->lin[(uint64_t)r * u->w + x] / 255.0f);
    } else {
        rows = 4;
        for (uint32_t bx = 0; bx < u->w / 4; bx++) {
            const uint8_t *b = u->lin + ((uint64_t)r * (u->w / 4) + bx) * 8;
            float r0 = b[0], r1 = b[1], pal[8] = {r0, r1};
            if (r0 > r1) for (int i = 1; i < 7; i++) pal[i + 1] = ((7 - i) * r0 + i * r1) / 7;
            else { for (int i = 1; i < 5; i++) pal[i + 1] = ((5 - i) * r0 + i * r1) / 5; pal[6] = 0, pal[7] = 255; }
            uint64_t bits = 0;
            for (int i = 0; i < 6; i++) bits |= (uint64_t)b[2 + i] << (8 * i);
            for (int i = 0; i < 16; i++) put(u, (uint64_t)(r * 4 + i / 4) * u->w + bx * 4 + i % 4, pal[(bits >> (3 * i)) & 7] / 255.0f);
        }
    }
    for (uint32_t l = 0; l < u->layers; l++)
        for (uint64_t y = (uint64_t)r * rows; y < (uint64_t)(r + 1) * rows; y++) {
            const float *a = u->a[l] + y * u->w;
            float *m = u->across[l] + y * u->w;
            for (uint32_t x = 0; x < u->w; x++) {
                float v = 0;
                for (uint32_t i = x >= 2 ? x - 2 : 0; i <= x + 2 && i < u->w; i++) v = a[i] > v ? a[i] : v;
                m[x] = v;
            }
        }
}

// The distance from a texel's centre to a straight edge crossing it with normal (gx, gy) and leaving it covered a:
// > 0 when the centre is outside (edtaa3's edgedf: the box-filtered coverage of a half plane, inverted)
static float edge_distance(float gx, float gy, float a) {
    if (gx == 0 || gy == 0) return 0.5f - a;
    float g = sqrtf(gx * gx + gy * gy);
    gx = fabsf(gx) / g, gy = fabsf(gy) / g;
    if (gx < gy) { float t = gx; gx = gy; gy = t; }
    float a1 = 0.5f * gy / gx;
    if (a < a1) return 0.5f * (gx + gy) - sqrtf(2 * gx * gy * a);
    if (a < 1 - a1) return (0.5f - a) * gx;
    return -0.5f * (gx + gy) + sqrtf(2 * gx * gy * (1 - a));
}

#define FAR 2.5f  // (more than any distance found within a texel's neighbours)

// Row y's distances: to the nearest edge among the texel and its 8 neighbours, each texel partly on the other side
// standing for an edge its coverage places (along the texel's gradient for itself, along the way to it for the
// others, as edtaa3 does). Interpolated, only distances under a texel matter, and those all have such a texel next to
// them.
static void distance_row(Up *u, uint32_t y) {
    const uint32_t w = u->w, h = u->h;
    for (uint32_t l = 0; l < u->layers; l++) {
        const float *a = u->a[l];
        Tx *t = u->t[l] + (uint64_t)y * w;
        uint8_t *flat = u->flat[l] + (uint64_t)y * w;
        for (uint32_t x = 0; x < w; x++) {
            float m = 0;  // the most coverage within 2 texels: a stroke if some texel is nearly covered
            for (uint32_t j = y >= 2 ? y - 2 : 0; j <= y + 2 && j < h; j++)
                if (u->across[l][(uint64_t)j * w + x] > m) m = u->across[l][(uint64_t)j * w + x];
            if (m == 0) {  // (most of a sheet)
                t[x] = (Tx){FAR, 0, 0};
                flat[x] = 0;
                continue;
            }
            float c[3][3];
            for (int j = 0; j < 3; j++)
                for (int i = 0; i < 3; i++) {
                    int64_t X = (int64_t)x + i - 1, Y = (int64_t)y + j - 1;
                    X = X < 0 ? 0 : X >= w ? w - 1 : X;
                    Y = Y < 0 ? 0 : Y >= h ? h - 1 : Y;
                    c[j][i] = a[(uint64_t)Y * w + X];
                }
            bool in = c[1][1] >= 0.5f, same = true;
            for (int j = 0; j < 3 && same; j++)
                for (int i = 0; i < 3 && same; i++) same = c[j][i] == (in ? 1 : 0);
            float best = FAR;
            for (int j = 0; j < 3 && !same; j++)
                for (int i = 0; i < 3; i++) {
                    float q = in ? 1 - c[j][i] : c[j][i];  // (how much of it is on the other side)
                    if (q <= 0) continue;
                    float d;
                    if (i == 1 && j == 1) {
                        const float r2 = 1.41421356f;
                        float gx = c[0][2] + r2 * c[1][2] + c[2][2] - c[0][0] - r2 * c[1][0] - c[2][0];
                        float gy = c[2][0] + r2 * c[2][1] + c[2][2] - c[0][0] - r2 * c[0][1] - c[0][2];
                        d = edge_distance(gx, gy, q);
                    } else if (i == 1 || j == 1) {
                        d = 1.5f - q;
                    } else {  // (diagonal: edge_distance(1, 1, q))
                        d = 1.41421356f + (q < 0.5f ? 0.70710678f - sqrtf(q) : sqrtf(1 - q) - 0.70710678f);
                    }
                    if (d < best) best = d;
                }
            float k = clampf((m - 0.55f) / 0.3f, 0, 1);
            t[x] = (Tx){in ? -best : best, k * k * (3 - 2 * k), c[1][1]};
            float q[4] = {c[1][1], c[1][2], c[2][1], c[2][2]};  // (the neighbours right, below, right below)
            flat[x] = q[0] == 0 && q[1] == 0 && q[2] == 0 && q[3] == 0 ? 0 : q[0] == 1 && q[1] == 1 && q[2] == 1 && q[3] == 1 ? 2 : 1;
        }
    }
}

// How wide an edge is in the new sheet's texels: SAVAGEWIND_FONT_EDGE (default 1.5). One texel (a pixel at the text's
// size on screen) made the outlined small print's band a hard, stair-stepped line; the game's own sheets have edges
// one old texel (n new ones) wide.
static float font_edge(void) {
    static float e = -1;
    if (e < 0) { const char *v = getenv("SAVAGEWIND_FONT_EDGE"); e = v && atof(v) > 0 ? (float)atof(v) : 1.5f; }
    return e;
}

// How far out the edge moves, in the old texels: SAVAGEWIND_FONT_BOLD (default 0.2). The game's soft edges spread a
// letter's ink about half a texel past its 50% line, which reads as weight; a sharp edge on that line looked thinner
// (item descriptions at 3x, 27 Oct: 0.2 matches the game's stroke weight, 0.35 is bolder).
static float font_bold(void) {
    static float b = -100;
    if (b == -100) { const char *v = getenv("SAVAGEWIND_FONT_BOLD"); b = v ? (float)atof(v) : 0.2f; }
    return b;
}

static uint8_t texel(const Up *u, uint32_t ox, uint32_t oy) {  // the new sheet's texel (ox, oy)
    uint64_t w = u->w, r0 = u->y0[oy] * w, r1 = u->y1[oy] * w;
    int x0 = u->x0[ox], x1 = u->x1[ox];
    float fx = u->fx[ox], fy = u->fy[oy], v = 0;
    float w00 = (1 - fx) * (1 - fy), w10 = fx * (1 - fy), w01 = (1 - fx) * fy, w11 = fx * fy;
    for (uint32_t l = 0; l < u->layers; l++) {
        uint8_t f = u->flat[l][r0 + x0];
        if (f != 1) { v += f / 2; continue; }
        const Tx *p = u->t[l] + r0 + x0, *q = u->t[l] + r0 + x1, *r = u->t[l] + r1 + x0, *z = u->t[l] + r1 + x1;
        float s = p->s * w00 + q->s * w10 + r->s * w01 + z->s * w11;
        float k = p->k * w00 + q->k * w10 + r->k * w01 + z->k * w11;
        float a = p->a * w00 + q->a * w10 + r->a * w01 + z->a * w11;
        float edge = clampf(0.5f - (s - font_bold()) * (float)u->n / font_edge(), 0, 1);
        v += clampf(a + k * (edge - a), 0, 1);
    }
    return (uint8_t)(v / u->layers * 255 + 0.5f);
}

// 16 values as a BC4 block: the better of its two modes (8 values between the extremes; 6 between the extremes
// other than 0 and 255, and those two)
static void bc4_block(const uint8_t v[16], uint8_t *b) {
    int lo = 255, hi = 0, lo6 = 255, hi6 = 0;
    for (int i = 0; i < 16; i++) {
        if (v[i] < lo) lo = v[i];
        if (v[i] > hi) hi = v[i];
        if (v[i] && v[i] != 255) {
            if (v[i] < lo6) lo6 = v[i];
            if (v[i] > hi6) hi6 = v[i];
        }
    }
    memset(b, 0, 8);
    b[0] = (uint8_t)hi, b[1] = (uint8_t)lo;
    if (hi == lo) return;
    if (lo6 > hi6) lo6 = hi6 = 0;  // (only 0 and 255)
    uint64_t bits[2] = {0, 0};
    float err[2] = {0, 0};
    for (int i = 0; i < 16; i++) {
        int t = (int)((float)(hi - v[i]) * 7 / (float)(hi - lo) + 0.5f);  // (8 values: hi = 0, then 2..7, lo = 1)
        float e = (float)(hi - v[i]) - (float)t * (float)(hi - lo) / 7;
        bits[0] |= (uint64_t)(t == 0 ? 0 : t == 7 ? 1 : t + 1) << (3 * i);
        err[0] += e * e;
        int idx = 0;  // (6 values: lo6 = 0, then 2..5, hi6 = 1; 0 = 6, 255 = 7)
        float p = (float)lo6;
        if (hi6 > lo6) {
            int t6 = (int)floorf((float)(v[i] - lo6) * 5 / (float)(hi6 - lo6) + 0.5f);
            t6 = t6 < 0 ? 0 : t6 > 5 ? 5 : t6;
            idx = t6 == 0 ? 0 : t6 == 5 ? 1 : t6 + 1;
            p = (float)lo6 + (float)t6 * (float)(hi6 - lo6) / 5;
        }
        float e6 = fabsf((float)v[i] - p);
        if ((float)v[i] < e6) e6 = (float)v[i], idx = 6;
        if ((float)(255 - v[i]) < e6) e6 = (float)(255 - v[i]), idx = 7;
        bits[1] |= (uint64_t)idx << (3 * i);
        err[1] += e6 * e6;
    }
    int m = err[1] < err[0];
    if (m) b[0] = (uint8_t)lo6, b[1] = (uint8_t)hi6;  // (r0 <= r1: the 6-value mode; r0 == r1 also reads so)
    for (int i = 0; i < 6; i++) b[2 + i] = (uint8_t)(bits[m] >> (8 * i));
}

static void output_row(Up *u, uint32_t r) {  // (BC4, one layer: a row of blocks)
    if (u->format == FMT_R8) {
        for (uint32_t x = 0; x < u->ow; x++) u->out[(uint64_t)r * u->ow + x] = texel(u, x, r);
        return;
    }
    int ya = u->y0[r * 4], yb = u->y0[r * 4 + 3];
    const uint8_t *flat = u->flat[0];
    for (uint32_t bx = 0; bx < u->ow / 4; bx++) {
        uint8_t *b = u->out + ((uint64_t)r * (u->ow / 4) + bx) * 8;
        int xa = u->x0[bx * 4], xb = u->x0[bx * 4 + 3];
        uint8_t f = flat[(uint64_t)ya * u->w + xa];  // (most blocks: all empty or all covered)
        for (int y = ya; y <= yb && f != 1; y++)
            for (int x = xa; x <= xb && f != 1; x++)
                if (flat[(uint64_t)y * u->w + x] != f) f = 1;
        if (f != 1) {
            memset(b, 0, 8);
            b[0] = b[1] = f ? 255 : 0;
            continue;
        }
        uint8_t v[16];
        for (int i = 0; i < 16; i++) v[i] = texel(u, bx * 4 + i % 4, r * 4 + i / 4);
        bc4_block(v, b);
    }
}

typedef struct { Up *u; void (*fn)(Up *, uint32_t); uint32_t r0, r1; } Band;
static void *band_run(void *p) {
    Band *b = p;
    for (uint32_t r = b->r0; r < b->r1; r++) b->fn(b->u, r);
    return NULL;
}

static void each_row(Up *u, void (*fn)(Up *, uint32_t), uint32_t rows) {  // rows 0..rows-1 on the cores
    enum { MAXT = 16 };
    int nt = SDL_GetNumLogicalCPUCores();
    nt = nt < 1 ? 1 : nt > MAXT ? MAXT : nt;
    if ((uint32_t)nt > rows) nt = (int)rows;
    pthread_t th[MAXT];
    bool started[MAXT] = {false};
    Band band[MAXT];
    for (int i = 0; i < nt; i++) {
        band[i] = (Band){u, fn, (uint32_t)((uint64_t)rows * i / nt), (uint32_t)((uint64_t)rows * (i + 1) / nt)};
        if (i) started[i] = pthread_create(&th[i], NULL, band_run, &band[i]) == 0;
    }
    band_run(&band[0]);
    for (int i = 1; i < nt; i++) {
        if (started[i]) pthread_join(th[i], NULL);
        else band_run(&band[i]);
    }
}

static void axis(uint32_t size, uint32_t n, int *i0, int *i1, float *f) {  // where each new texel falls between two
    for (uint32_t o = 0; o < size * n; o++) {
        float p = (o + 0.5f) / (float)n - 0.5f, fl = floorf(p);
        int lo = (int)fl;
        f[o] = p - fl;
        i0[o] = lo < 0 ? 0 : lo >= (int)size ? (int)size - 1 : lo;
        i1[o] = lo + 1 < 0 ? 0 : lo + 1 >= (int)size ? (int)size - 1 : lo + 1;
    }
}

// Made sheets are kept in the cache folder (fonts/<key>.bin) and read back in later sessions: making them took 70-340
// ms each at 3x while the game loads (27 Oct, 5120x2160: ~0.6 s of a stopped title screen); a read is a few ms. The key
// covers the sheet's texels, its shape, the scale and the edge settings (FONT_SHEET_VERSION: the making itself).
#define FONT_SHEET_VERSION 1
const char *vk_cache_dir(void);
static bool sheet_cache_path(char *path, size_t size, uint32_t format, const uint8_t *lin, uint64_t in_bytes, uint32_t w,
                             uint32_t h, int n) {
    uint64_t k = sheet_hash(lin, in_bytes);
    float e = font_edge(), b = font_bold();
    uint32_t eb[2];
    memcpy(&eb[0], &e, 4), memcpy(&eb[1], &b, 4);
    k = (k ^ ((uint64_t)format << 48 | (uint64_t)w << 24 | h)) * 0x100000001b3ull;
    k = (k ^ ((uint64_t)eb[0] << 32 | eb[1])) * 0x100000001b3ull;
    k = (k ^ ((uint64_t)n << 8 | FONT_SHEET_VERSION)) * 0x100000001b3ull;
    char dir[512];
    snprintf(dir, sizeof dir, "%s/fonts", vk_cache_dir());
    mkdir(dir, 0755);
    return snprintf(path, size, "%s/%016llx.bin", dir, (unsigned long long)k) < (int)size;
}

static bool sheet_make(uint32_t format, const uint8_t *lin, uint32_t w, uint32_t h, int n, uint8_t *out);
void font_sheet_upscale(uint32_t format, const uint8_t *lin, uint32_t w, uint32_t h, int n, uint8_t *dst) {
    uint64_t t0 = SDL_GetTicksNS(), ow = (uint64_t)w * (uint32_t)n, oh = (uint64_t)h * (uint32_t)n;
    uint64_t in_bytes = format == FMT_R8 ? (uint64_t)w * h : (uint64_t)w * h / 2, out_bytes = format == FMT_R8 ? ow * oh : ow * oh / 2;
    // (dst is upload memory, often write-combined over the PCIe bus: reading it back to write the file crawled - a
    // second a sheet - so the sheet is made or read in ordinary memory and copied there once)
    uint8_t *out = malloc(out_bytes);
    if (!out) { sheet_make(format, lin, w, h, n, dst); return; }
    char path[600];
    bool cached = false;
    if (sheet_cache_path(path, sizeof path, format, lin, in_bytes, w, h, n)) {
        FILE *f = fopen(path, "rb");
        if (f) {
            cached = fread(out, 1, out_bytes, f) == out_bytes && fgetc(f) == EOF;
            fclose(f);
        }
        if (!cached && sheet_make(format, lin, w, h, n, out)) {  // (an empty one, out of memory: not kept)
            char tmp[640];  // (this thread's: two may make the same sheet at once; the second rename fails, harmlessly)
            snprintf(tmp, sizeof tmp, "%s.%llu.tmp", path, (unsigned long long)SDL_GetCurrentThreadID());
            f = fopen(tmp, "wb");
            if (f) {
                bool written = fwrite(out, 1, out_bytes, f) == out_bytes;
                written = fclose(f) == 0 && written;
                if (!written || rename(tmp, path) != 0) remove(tmp);
            }
        }
    } else {
        sheet_make(format, lin, w, h, n, out);
    }
    uint64_t t1 = SDL_GetTicksNS();
    memcpy(dst, out, out_bytes);
    free(out);
    static int nlog;
    if (nlog++ < 48) LOG("[fonts] %ux%u sheet %s %dx in %.1f ms (+%.1f ms into upload memory)\n", w, h, cached ? "read" : "made", n,
                         (t1 - t0) / 1e6, (SDL_GetTicksNS() - t1) / 1e6);
}

static bool sheet_make(uint32_t format, const uint8_t *lin, uint32_t w, uint32_t h, int n, uint8_t *out) {
    Up u = {.format = format, .w = w, .h = h, .n = (uint32_t)n, .ow = w * (uint32_t)n, .oh = h * (uint32_t)n,
            .layers = format == FMT_R8 ? 2 : 1, .lin = lin, .out = out};
    uint64_t px = (uint64_t)w * h;
    bool ok = true;
    for (uint32_t l = 0; l < u.layers; l++) {
        u.a[l] = malloc(px * sizeof(float)), u.across[l] = malloc(px * sizeof(float)), u.t[l] = malloc(px * sizeof(Tx));
        u.flat[l] = malloc(px);
        ok = ok && u.a[l] && u.across[l] && u.t[l] && u.flat[l];
    }
    u.x0 = malloc(u.ow * sizeof(int)), u.x1 = malloc(u.ow * sizeof(int)), u.fx = malloc(u.ow * sizeof(float));
    u.y0 = malloc(u.oh * sizeof(int)), u.y1 = malloc(u.oh * sizeof(int)), u.fy = malloc(u.oh * sizeof(float));
    ok = ok && u.x0 && u.x1 && u.fx && u.y0 && u.y1 && u.fy;
    if (ok) {
        uint32_t blocks = format == FMT_R8 ? 1 : 4;
        axis(w, u.n, u.x0, u.x1, u.fx);
        axis(h, u.n, u.y0, u.y1, u.fy);
        each_row(&u, decode_row, h / blocks);
        each_row(&u, distance_row, h);
        each_row(&u, output_row, u.oh / blocks);
    } else {
        memset(out, 0, format == FMT_R8 ? (uint64_t)u.ow * u.oh : (uint64_t)u.ow * u.oh / 2);
    }
    for (uint32_t l = 0; l < u.layers; l++) free(u.a[l]), free(u.across[l]), free(u.t[l]), free(u.flat[l]);
    free(u.x0), free(u.x1), free(u.fx), free(u.y0), free(u.y1), free(u.fy);
    if (!ok) LOG("[fonts] %ux%u sheet at %dx: out of memory, left empty\n", w, h, n);
    return ok;
}
