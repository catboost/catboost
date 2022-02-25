#include "common.h"

namespace {
    #include "farmhashte.cc"
}

namespace farmhashnt {
#if !can_use_sse41 || !x86_64

uint32_t Hash32(const char *s, size_t len) {
  FARMHASH_DIE_IF_MISCONFIGURED;
  return s == NULL ? 0 : len;
}

uint32_t Hash32WithSeed(const char *s, size_t len, uint32_t seed) {
  FARMHASH_DIE_IF_MISCONFIGURED;
  return seed + Hash32(s, len);
}

#else

uint32_t Hash32(const char *s, size_t len) {
  return static_cast<uint32_t>(farmhashte::Hash64(s, len));
}

uint32_t Hash32WithSeed(const char *s, size_t len, uint32_t seed) {
  return static_cast<uint32_t>(farmhashte::Hash64WithSeed(s, len, seed));
}

#endif
}  // namespace farmhashnt
