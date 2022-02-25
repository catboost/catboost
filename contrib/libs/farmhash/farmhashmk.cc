#include "common.h"

namespace {
    #include "farmhashnt.cc"
}

namespace farmhashmk {
#undef Fetch
#define Fetch Fetch32

#undef Rotate
#define Rotate Rotate32

#undef Bswap
#define Bswap Bswap32

STATIC_INLINE uint32_t Hash32Len13to24(const char *s, size_t len, uint32_t seed = 0) {
  uint32_t a = Fetch(s - 4 + (len >> 1));
  uint32_t b = Fetch(s + 4);
  uint32_t c = Fetch(s + len - 8);
  uint32_t d = Fetch(s + (len >> 1));
  uint32_t e = Fetch(s);
  uint32_t f = Fetch(s + len - 4);
  uint32_t h = d * c1 + len + seed;
  a = Rotate(a, 12) + f;
  h = Mur(c, h) + a;
  a = Rotate(a, 3) + c;
  h = Mur(e, h) + a;
  a = Rotate(a + f, 12) + d;
  h = Mur(b ^ seed, h) + a;
  return fmix(h);
}

STATIC_INLINE uint32_t Hash32Len0to4(const char *s, size_t len, uint32_t seed = 0) {
  uint32_t b = seed;
  uint32_t c = 9;
  for (size_t i = 0; i < len; i++) {
    signed char v = s[i];
    b = b * c1 + v;
    c ^= b;
  }
  return fmix(Mur(b, Mur(len, c)));
}

STATIC_INLINE uint32_t Hash32Len5to12(const char *s, size_t len, uint32_t seed = 0) {
  uint32_t a = len, b = len * 5, c = 9, d = b + seed;
  a += Fetch(s);
  b += Fetch(s + len - 4);
  c += Fetch(s + ((len >> 1) & 4));
  return fmix(seed ^ Mur(c, Mur(b, Mur(a, d))));
}

uint32_t Hash32(const char *s, size_t len) {
  if (len <= 24) {
    return len <= 12 ?
        (len <= 4 ? Hash32Len0to4(s, len) : Hash32Len5to12(s, len)) :
        Hash32Len13to24(s, len);
  }

  // len > 24
  uint32_t h = len, g = c1 * len, f = g;
  uint32_t a0 = Rotate(Fetch(s + len - 4) * c1, 17) * c2;
  uint32_t a1 = Rotate(Fetch(s + len - 8) * c1, 17) * c2;
  uint32_t a2 = Rotate(Fetch(s + len - 16) * c1, 17) * c2;
  uint32_t a3 = Rotate(Fetch(s + len - 12) * c1, 17) * c2;
  uint32_t a4 = Rotate(Fetch(s + len - 20) * c1, 17) * c2;
  h ^= a0;
  h = Rotate(h, 19);
  h = h * 5 + 0xe6546b64;
  h ^= a2;
  h = Rotate(h, 19);
  h = h * 5 + 0xe6546b64;
  g ^= a1;
  g = Rotate(g, 19);
  g = g * 5 + 0xe6546b64;
  g ^= a3;
  g = Rotate(g, 19);
  g = g * 5 + 0xe6546b64;
  f += a4;
  f = Rotate(f, 19) + 113;
  size_t iters = (len - 1) / 20;
  do {
    uint32_t a = Fetch(s);
    uint32_t b = Fetch(s + 4);
    uint32_t c = Fetch(s + 8);
    uint32_t d = Fetch(s + 12);
    uint32_t e = Fetch(s + 16);
    h += a;
    g += b;
    f += c;
    h = Mur(d, h) + e;
    g = Mur(c, g) + a;
    f = Mur(b + e * c1, f) + d;
    f += g;
    g += f;
    s += 20;
  } while (--iters != 0);
  g = Rotate(g, 11) * c1;
  g = Rotate(g, 17) * c1;
  f = Rotate(f, 11) * c1;
  f = Rotate(f, 17) * c1;
  h = Rotate(h + g, 19);
  h = h * 5 + 0xe6546b64;
  h = Rotate(h, 17) * c1;
  h = Rotate(h + f, 19);
  h = h * 5 + 0xe6546b64;
  h = Rotate(h, 17) * c1;
  return h;
}

uint32_t Hash32WithSeed(const char *s, size_t len, uint32_t seed) {
  if (len <= 24) {
    if (len >= 13) return Hash32Len13to24(s, len, seed * c1);
    else if (len >= 5) return Hash32Len5to12(s, len, seed);
    else return Hash32Len0to4(s, len, seed);
  }
  uint32_t h = Hash32Len13to24(s, 24, seed ^ len);
  return Mur(Hash32(s + 24, len - 24) + seed, h);
}
}  // namespace farmhashmk
