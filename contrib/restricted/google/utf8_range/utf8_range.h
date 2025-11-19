#ifndef THIRD_PARTY_UTF8_RANGE_UTF8_RANGE_H_
#define THIRD_PARTY_UTF8_RANGE_UTF8_RANGE_H_

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Returns 1 if the sequence of characters is a valid UTF-8 sequence, otherwise
// 0.
int utf8_range_IsValid(const char* data, size_t len);

// Returns the length in bytes of the prefix of str that is all
// structurally valid UTF-8.
size_t utf8_range_ValidPrefix(const char* data, size_t len);

// Legacy API for backward compatibility
// Only needed for platforms without SSE4.1 (where range2-sse.c is not compiled)
#ifndef __SSE4_1__
int utf8_naive(const unsigned char* data, int len);
static inline int utf8_range2(const unsigned char* data, int len) {
  return utf8_naive(data, len);
}
#endif

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // THIRD_PARTY_UTF8_RANGE_UTF8_RANGE_H_
