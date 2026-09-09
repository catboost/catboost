#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#define IBDRV_NOEXCEPT noexcept
#else
#define IBDRV_NOEXCEPT
#endif

// Loads the host libibverbs and verifies the symbols required by a caller.
// Returns 0 on success or a negative errno value. Unlike the regular ibdrv
// wrappers, this function never lets a C++ exception escape to the caller.
int ibdrv_try_load_ibverbs(
    const char* const* required_symbols,
    size_t required_symbol_count,
    char* error_message,
    size_t error_message_size) IBDRV_NOEXCEPT;

#ifdef __cplusplus
}
#endif

#undef IBDRV_NOEXCEPT
