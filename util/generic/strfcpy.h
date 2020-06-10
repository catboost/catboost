#pragma once

/*
 * A faster version of strncpy()
 *
 * Comparison with other str*cpy():
 *   strcpy()  - buffer overflow ready
 *   strncpy() - wastes time filling exactly n bytes with 0
 *   strlcpy() - wastes time searching for the length of src
 *   memcpy()  - wastes time copying exactly n bytes even if the string is shorter
 *
 * WARN: these words are not true. strfcpy MUST BE replaced with OpenBSD version of strlcpy.
 */

#include <stddef.h>

// Copy src to string dst of size siz.  At most siz-1 characters
// will be copied.  Always NUL terminates (unless siz == 0).
void strfcpy(char* dst, const char* src, size_t n);
