#pragma once
#include <stddef.h>
#include <stdint.h>

// Replace statically indexed c1 reads with their exact compiler-constant bits.
// Returns a malloc'd source, or NULL when the source needs the original binding.
// Dynamically indexed c1, unknown syntax and reads outside the declared blob fall
// back; no floating-point conversions or algebraic approximations are performed.
// The specialization cache also hashes this implementation and the complete blob.
char *vk_shader_literal_constants(const char *source, const uint8_t *data, size_t size);
