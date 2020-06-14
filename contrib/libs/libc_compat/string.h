#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

size_t strlcpy(char* dst, const char* src, size_t len);
size_t strlcat(char* dst, const char* src, size_t len);

#ifdef __cplusplus
} //extern "C"
#endif
