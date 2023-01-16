#include "common.h"

namespace {
    #include "farmhashsa.cc"
}

namespace farmhashcc {
// This file provides a 32-bit hash equivalent to CityHash32 (v1.1.1)
// and a 128-bit hash equivalent to CityHash128 (v1.1.1).  It also provides
// a seeded 32-bit hash function similar to CityHash32.

#undef Fetch
#define Fetch Fetch32

#undef Rotate
#define Rotate Rotate32

#undef Bswap
#define Bswap Bswap32

STATIC_INLINE uint32_t Hash32Len13to24(const char *s, size_t len) {
  uint32_t a = Fetch(s - 4 + (len >> 1));
  uint32_t b = Fetch(s + 4);
  uint32_t c = Fetch(s + len - 8);
  uint32_t d = Fetch(s + (len >> 1));
  uint32_t e = Fetch(s);
  uint32_t f = Fetch(s + len - 4);
  uint32_t h = len;

  return fmix(Mur(f, Mur(e, Mur(d, Mur(c, Mur(b, Mur(a, h)))))));
}

STATIC_INLINE uint32_t Hash32Len0to4(const char *s, size_t len) {
  uint32_t b = 0;
  uint32_t c = 9;
  for (size_t i = 0; i < len; i++) {
    signed char v = s[i];
    b = b * c1 + v;
    c ^= b;
  }
  return fmix(Mur(b, Mur(len, c)));
}

STATIC_INLINE uint32_t Hash32Len5to12(const char *s, size_t len) {
  uint32_t a = len, b = len * 5, c = 9, d = b;
  a += Fetch(s);
  b += Fetch(s + len - 4);
  c += Fetch(s + ((len >> 1) & 4));
  return fmix(Mur(c, Mur(b, Mur(a, d))));
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
  f = Rotate(f, 19);
  f = f * 5 + 0xe6546b64;
  size_t iters = (len - 1) / 20;
  do {
    uint32_t a0 = Rotate(Fetch(s) * c1, 17) * c2;
    uint32_t a1 = Fetch(s + 4);
    uint32_t a2 = Rotate(Fetch(s + 8) * c1, 17) * c2;
    uint32_t a3 = Rotate(Fetch(s + 12) * c1, 17) * c2;
    uint32_t a4 = Fetch(s + 16);
    h ^= a0;
    h = Rotate(h, 18);
    h = h * 5 + 0xe6546b64;
    f += a1;
    f = Rotate(f, 19);
    f = f * c1;
    g += a2;
    g = Rotate(g, 18);
    g = g * 5 + 0xe6546b64;
    h ^= a3 + a1;
    h = Rotate(h, 19);
    h = h * 5 + 0xe6546b64;
    g ^= a4;
    g = Bswap(g) * 5;
    h += a4 * 5;
    h = Bswap(h);
    f += a0;
    PERMUTE3(f, h, g);
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
    if (len >= 13) return farmhashmk::Hash32Len13to24(s, len, seed * c1);
    else if (len >= 5) return farmhashmk::Hash32Len5to12(s, len, seed);
    else return farmhashmk::Hash32Len0to4(s, len, seed);
  }
  uint32_t h = farmhashmk::Hash32Len13to24(s, 24, seed ^ len);
  return Mur(Hash32(s + 24, len - 24) + seed, h);
}

#undef Fetch
#define Fetch Fetch64

#undef Rotate
#define Rotate Rotate64

#undef Bswap
#define Bswap Bswap64

STATIC_INLINE uint64_t ShiftMix(uint64_t val) {
  return val ^ (val >> 47);
}

STATIC_INLINE uint64_t HashLen16(uint64_t u, uint64_t v) {
  return Hash128to64(Uint128(u, v));
}

STATIC_INLINE uint64_t HashLen16(uint64_t u, uint64_t v, uint64_t mul) {
  // Murmur-inspired hashing.
  uint64_t a = (u ^ v) * mul;
  a ^= (a >> 47);
  uint64_t b = (v ^ a) * mul;
  b ^= (b >> 47);
  b *= mul;
  return b;
}

STATIC_INLINE uint64_t HashLen0to16(const char *s, size_t len) {
  if (len >= 8) {
    uint64_t mul = k2 + len * 2;
    uint64_t a = Fetch(s) + k2;
    uint64_t b = Fetch(s + len - 8);
    uint64_t c = Rotate(b, 37) * mul + a;
    uint64_t d = (Rotate(a, 25) + b) * mul;
    return HashLen16(c, d, mul);
  }
  if (len >= 4) {
    uint64_t mul = k2 + len * 2;
    uint64_t a = Fetch32(s);
    return HashLen16(len + (a << 3), Fetch32(s + len - 4), mul);
  }
  if (len > 0) {
    uint8_t a = s[0];
    uint8_t b = s[len >> 1];
    uint8_t c = s[len - 1];
    uint32_t y = static_cast<uint32_t>(a) + (static_cast<uint32_t>(b) << 8);
    uint32_t z = len + (static_cast<uint32_t>(c) << 2);
    return ShiftMix(y * k2 ^ z * k0) * k2;
  }
  return k2;
}

// Return a 16-byte hash for 48 bytes.  Quick and dirty.
// Callers do best to use "random-looking" values for a and b.
STATIC_INLINE pair<uint64_t, uint64_t> WeakHashLen32WithSeeds(
    uint64_t w, uint64_t x, uint64_t y, uint64_t z, uint64_t a, uint64_t b) {
  a += w;
  b = Rotate(b + a + z, 21);
  uint64_t c = a;
  a += x;
  a += y;
  b += Rotate(a, 44);
  return make_pair(a + z, b + c);
}

// Return a 16-byte hash for s[0] ... s[31], a, and b.  Quick and dirty.
STATIC_INLINE pair<uint64_t, uint64_t> WeakHashLen32WithSeeds(
    const char* s, uint64_t a, uint64_t b) {
  return WeakHashLen32WithSeeds(Fetch(s),
                                Fetch(s + 8),
                                Fetch(s + 16),
                                Fetch(s + 24),
                                a,
                                b);
}



// A subroutine for CityHash128().  Returns a decent 128-bit hash for strings
// of any length representable in signed long.  Based on City and Murmur.
STATIC_INLINE uint128_t CityMurmur(const char *s, size_t len, uint128_t seed) {
  uint64_t a = Uint128Low64(seed);
  uint64_t b = Uint128High64(seed);
  uint64_t c = 0;
  uint64_t d = 0;
  signed long l = len - 16;
  if (l <= 0) {  // len <= 16
    a = ShiftMix(a * k1) * k1;
    c = b * k1 + HashLen0to16(s, len);
    d = ShiftMix(a + (len >= 8 ? Fetch(s) : c));
  } else {  // len > 16
    c = HashLen16(Fetch(s + len - 8) + k1, a);
    d = HashLen16(b + len, c + Fetch(s + len - 16));
    a += d;
    do {
      a ^= ShiftMix(Fetch(s) * k1) * k1;
      a *= k1;
      b ^= a;
      c ^= ShiftMix(Fetch(s + 8) * k1) * k1;
      c *= k1;
      d ^= c;
      s += 16;
      l -= 16;
    } while (l > 0);
  }
  a = HashLen16(a, c);
  b = HashLen16(d, b);
  return Uint128(a ^ b, HashLen16(b, a));
}

uint128_t CityHash128WithSeed(const char *s, size_t len, uint128_t seed) {
  if (len < 128) {
    return CityMurmur(s, len, seed);
  }

  // We expect len >= 128 to be the common case.  Keep 56 bytes of state:
  // v, w, x, y, and z.
  pair<uint64_t, uint64_t> v, w;
  uint64_t x = Uint128Low64(seed);
  uint64_t y = Uint128High64(seed);
  uint64_t z = len * k1;
  v.first = Rotate(y ^ k1, 49) * k1 + Fetch(s);
  v.second = Rotate(v.first, 42) * k1 + Fetch(s + 8);
  w.first = Rotate(y + z, 35) * k1 + x;
  w.second = Rotate(x + Fetch(s + 88), 53) * k1;

  // This is the same inner loop as CityHash64(), manually unrolled.
  do {
    x = Rotate(x + y + v.first + Fetch(s + 8), 37) * k1;
    y = Rotate(y + v.second + Fetch(s + 48), 42) * k1;
    x ^= w.second;
    y += v.first + Fetch(s + 40);
    z = Rotate(z + w.first, 33) * k1;
    v = WeakHashLen32WithSeeds(s, v.second * k1, x + w.first);
    w = WeakHashLen32WithSeeds(s + 32, z + w.second, y + Fetch(s + 16));
    std::swap(z, x);
    s += 64;
    x = Rotate(x + y + v.first + Fetch(s + 8), 37) * k1;
    y = Rotate(y + v.second + Fetch(s + 48), 42) * k1;
    x ^= w.second;
    y += v.first + Fetch(s + 40);
    z = Rotate(z + w.first, 33) * k1;
    v = WeakHashLen32WithSeeds(s, v.second * k1, x + w.first);
    w = WeakHashLen32WithSeeds(s + 32, z + w.second, y + Fetch(s + 16));
    std::swap(z, x);
    s += 64;
    len -= 128;
  } while (LIKELY(len >= 128));
  x += Rotate(v.first + z, 49) * k0;
  y = y * k0 + Rotate(w.second, 37);
  z = z * k0 + Rotate(w.first, 27);
  w.first *= 9;
  v.first *= k0;
  // If 0 < len < 128, hash up to 4 chunks of 32 bytes each from the end of s.
  for (size_t tail_done = 0; tail_done < len; ) {
    tail_done += 32;
    y = Rotate(x + y, 42) * k0 + v.second;
    w.first += Fetch(s + len - tail_done + 16);
    x = x * k0 + w.first;
    z += w.second + Fetch(s + len - tail_done);
    w.second += v.first;
    v = WeakHashLen32WithSeeds(s + len - tail_done, v.first + z, v.second);
    v.first *= k0;
  }
  // At this point our 56 bytes of state should contain more than
  // enough information for a strong 128-bit hash.  We use two
  // different 56-byte-to-8-byte hashes to get a 16-byte final result.
  x = HashLen16(x, v.first);
  y = HashLen16(y + z, w.first);
  return Uint128(HashLen16(x + v.second, w.second) + y,
                 HashLen16(x + w.second, y + v.second));
}

STATIC_INLINE uint128_t CityHash128(const char *s, size_t len) {
  return len >= 16 ?
      CityHash128WithSeed(s + 16, len - 16,
                          Uint128(Fetch(s), Fetch(s + 8) + k0)) :
      CityHash128WithSeed(s, len, Uint128(k0, k1));
}

uint128_t Fingerprint128(const char* s, size_t len) {
  return CityHash128(s, len);
}
}  // namespace farmhashcc
