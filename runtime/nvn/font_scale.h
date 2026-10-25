// Sharper bitmap text (runtime/nvn/font_scale.c): the game's font sheets made larger at upload.
#pragma once
#include <stdbool.h>
#include <stdint.h>

// SAVAGEWIND_FONT_SCALE: how many times as large font sheets are made, 2..4; 1 (the default): off; 0 ("auto"): the
// caller works it out from the output size.
int font_scale_setting(void);

// Whether the texture whose storage is at mem (layers of layer_bytes; NVN format, w x h texels, levels) holds the
// game's font sheets, in every layer.
bool font_sheet_is(const uint8_t *mem, uint64_t layer_bytes, uint32_t layers, uint32_t format, uint32_t w, uint32_t h,
                   uint32_t levels);

// One sheet as uploaded (lin: its texels in linear rows, 4x4 BC4 blocks or R8; w x h texels, multiples of 4) -> out,
// n times as wide and as high in the same format.
void font_sheet_upscale(uint32_t format, const uint8_t *lin, uint32_t w, uint32_t h, int n, uint8_t *out);
