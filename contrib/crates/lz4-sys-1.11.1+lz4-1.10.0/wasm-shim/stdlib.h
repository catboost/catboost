#include <stddef.h>

#ifndef _STDLIB_H
#define _STDLIB_H 1

void *rust_lz4_wasm_shim_malloc(size_t size);
void *rust_lz4_wasm_shim_calloc(size_t nmemb, size_t size);
void rust_lz4_wasm_shim_free(void *ptr);

#define malloc(size) rust_lz4_wasm_shim_malloc(size)
#define calloc(nmemb, size) rust_lz4_wasm_shim_calloc(nmemb, size)
#define free(ptr) rust_lz4_wasm_shim_free(ptr)

#endif // _STDLIB_H
