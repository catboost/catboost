#include "common.h"

namespace {
    #include "farmhashuo.cc"
}

namespace farmhashxo {
#undef Fetch
#define Fetch Fetch64

#undef Rotate
#define Rotate Rotate64

STATIC_INLINE uint64_t H32(const char *s, size_t len, uint64_t mul,
                           uint64_t seed0 = 0, uint64_t seed1 = 0) {
  uint64_t a = Fetch(s) * k1;
  uint64_t b = Fetch(s + 8);
  uint64_t c = Fetch(s + len - 8) * mul;
  uint64_t d = Fetch(s + len - 16) * k2;
  uint64_t u = Rotate(a + b, 43) + Rotate(c, 30) + d + seed0;
  uint64_t v = a + Rotate(b + k2, 18) + c + seed1;
  a = farmhashna::ShiftMix((u ^ v) * mul);
  b = farmhashna::ShiftMix((v ^ a) * mul);
  return b;
}

// Return an 8-byte hash for 33 to 64 bytes.
STATIC_INLINE uint64_t HashLen33to64(const char *s, size_t len) {
  uint64_t mul0 = k2 - 30;
  uint64_t mul1 = k2 - 30 + 2 * len;
  uint64_t h0 = H32(s, 32, mul0);
  uint64_t h1 = H32(s + len - 32, 32, mul1);
  return ((h1 * mul1) + h0) * mul1;
}

// Return an 8-byte hash for 65 to 96 bytes.
STATIC_INLINE uint64_t HashLen65to96(const char *s, size_t len) {
  uint64_t mul0 = k2 - 114;
  uint64_t mul1 = k2 - 114 + 2 * len;
  uint64_t h0 = H32(s, 32, mul0);
  uint64_t h1 = H32(s + 32, 32, mul1);
  uint64_t h2 = H32(s + len - 32, 32, mul1, h0, h1);
  return (h2 * 9 + (h0 >> 17) + (h1 >> 21)) * mul1;
}

uint64_t Hash64(const char *s, size_t len) {
  if (len <= 32) {
    if (len <= 16) {
      return farmhashna::HashLen0to16(s, len);
    } else {
      return farmhashna::HashLen17to32(s, len);
    }
  } else if (len <= 64) {
    return HashLen33to64(s, len);
  } else if (len <= 96) {
    return HashLen65to96(s, len);
  } else if (len <= 256) {
    return farmhashna::Hash64(s, len);
  } else {
    return farmhashuo::Hash64(s, len);
  }
}

uint64_t Hash64WithSeeds(const char *s, size_t len, uint64_t seed0, uint64_t seed1) {
  return farmhashuo::Hash64WithSeeds(s, len, seed0, seed1);
}

uint64_t Hash64WithSeed(const char *s, size_t len, uint64_t seed) {
  return farmhashuo::Hash64WithSeed(s, len, seed);
}
}  // namespace farmhashxo
