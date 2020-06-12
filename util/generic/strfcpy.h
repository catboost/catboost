#pragma once

/*
 * strfcpy is a faster version of strlcpy().
 * It returns void thus does not wastes time computing
 * (most likely, unneeded) strlen(str)
 *
 * Comparison with other copying functions:
 *   strcpy()  - buffer overflow ready
 *   strncpy() - wastes time filling exactly n bytes with 0
 *   strlcpy() - wastes time searching for the length of src
 *   memcpy()  - wastes time copying exactly n bytes even if the string is shorter
 */

#include <stddef.h>

void strfcpy(char* dst, const char* src, size_t n);
