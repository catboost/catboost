#include "common.h"

namespace {
    #include "farmhashxo.cc"
}

namespace farmhashte {
#if !can_use_sse41 || !x86_64

uint64_t Hash64(const char *s, size_t len) {
  FARMHASH_DIE_IF_MISCONFIGURED;
  return s == NULL ? 0 : len;
}

uint64_t Hash64WithSeed(const char *s, size_t len, uint64_t seed) {
  FARMHASH_DIE_IF_MISCONFIGURED;
  return seed + Hash64(s, len);
}

uint64_t Hash64WithSeeds(const char *s, size_t len,
                         uint64_t seed0, uint64_t seed1) {
  FARMHASH_DIE_IF_MISCONFIGURED;
  return seed0 + seed1 + Hash64(s, len);
}

#else

#undef Fetch
#define Fetch Fetch64

#undef Rotate
#define Rotate Rotate64

#undef Bswap
#define Bswap Bswap64

// Helpers for data-parallel operations (1x 128 bits or 2x 64 or 4x 32).
STATIC_INLINE __m128i Add(__m128i x, __m128i y) { return _mm_add_epi64(x, y); }
STATIC_INLINE __m128i Xor(__m128i x, __m128i y) { return _mm_xor_si128(x, y); }
STATIC_INLINE __m128i Mul(__m128i x, __m128i y) { return _mm_mullo_epi32(x, y); }
STATIC_INLINE __m128i Shuf(__m128i x, __m128i y) { return _mm_shuffle_epi8(y, x); }

// Requires n >= 256.  Requires SSE4.1. Should be slightly faster if the
// compiler uses AVX instructions (e.g., use the -mavx flag with GCC).
STATIC_INLINE uint64_t Hash64Long(const char* s, size_t n,
                                  uint64_t seed0, uint64_t seed1) {
  const __m128i kShuf =
      _mm_set_epi8(4, 11, 10, 5, 8, 15, 6, 9, 12, 2, 14, 13, 0, 7, 3, 1);
  const __m128i kMult =
      _mm_set_epi8(0xbd, 0xd6, 0x33, 0x39, 0x45, 0x54, 0xfa, 0x03,
                   0x34, 0x3e, 0x33, 0xed, 0xcc, 0x9e, 0x2d, 0x51);
  uint64_t seed2 = (seed0 + 113) * (seed1 + 9);
  uint64_t seed3 = (Rotate(seed0, 23) + 27) * (Rotate(seed1, 30) + 111);
  __m128i d0 = _mm_cvtsi64_si128(seed0);
  __m128i d1 = _mm_cvtsi64_si128(seed1);
  __m128i d2 = Shuf(kShuf, d0);
  __m128i d3 = Shuf(kShuf, d1);
  __m128i d4 = Xor(d0, d1);
  __m128i d5 = Xor(d1, d2);
  __m128i d6 = Xor(d2, d4);
  __m128i d7 = _mm_set1_epi32(seed2 >> 32);
  __m128i d8 = Mul(kMult, d2);
  __m128i d9 = _mm_set1_epi32(seed3 >> 32);
  __m128i d10 = _mm_set1_epi32(seed3);
  __m128i d11 = Add(d2, _mm_set1_epi32(seed2));
  const char* end = s + (n & ~static_cast<size_t>(255));
  do {
    __m128i z;
    z = Fetch128(s);
    d0 = Add(d0, z);
    d1 = Shuf(kShuf, d1);
    d2 = Xor(d2, d0);
    d4 = Xor(d4, z);
    d4 = Xor(d4, d1);
    std::swap(d0, d6);
    z = Fetch128(s + 16);
    d5 = Add(d5, z);
    d6 = Shuf(kShuf, d6);
    d8 = Shuf(kShuf, d8);
    d7 = Xor(d7, d5);
    d0 = Xor(d0, z);
    d0 = Xor(d0, d6);
    std::swap(d5, d11);
    z = Fetch128(s + 32);
    d1 = Add(d1, z);
    d2 = Shuf(kShuf, d2);
    d4 = Shuf(kShuf, d4);
    d5 = Xor(d5, z);
    d5 = Xor(d5, d2);
    std::swap(d10, d4);
    z = Fetch128(s + 48);
    d6 = Add(d6, z);
    d7 = Shuf(kShuf, d7);
    d0 = Shuf(kShuf, d0);
    d8 = Xor(d8, d6);
    d1 = Xor(d1, z);
    d1 = Add(d1, d7);
    z = Fetch128(s + 64);
    d2 = Add(d2, z);
    d5 = Shuf(kShuf, d5);
    d4 = Add(d4, d2);
    d6 = Xor(d6, z);
    d6 = Xor(d6, d11);
    std::swap(d8, d2);
    z = Fetch128(s + 80);
    d7 = Xor(d7, z);
    d8 = Shuf(kShuf, d8);
    d1 = Shuf(kShuf, d1);
    d0 = Add(d0, d7);
    d2 = Add(d2, z);
    d2 = Add(d2, d8);
    std::swap(d1, d7);
    z = Fetch128(s + 96);
    d4 = Shuf(kShuf, d4);
    d6 = Shuf(kShuf, d6);
    d8 = Mul(kMult, d8);
    d5 = Xor(d5, d11);
    d7 = Xor(d7, z);
    d7 = Add(d7, d4);
    std::swap(d6, d0);
    z = Fetch128(s + 112);
    d8 = Add(d8, z);
    d0 = Shuf(kShuf, d0);
    d2 = Shuf(kShuf, d2);
    d1 = Xor(d1, d8);
    d10 = Xor(d10, z);
    d10 = Xor(d10, d0);
    std::swap(d11, d5);
    z = Fetch128(s + 128);
    d4 = Add(d4, z);
    d5 = Shuf(kShuf, d5);
    d7 = Shuf(kShuf, d7);
    d6 = Add(d6, d4);
    d8 = Xor(d8, z);
    d8 = Xor(d8, d5);
    std::swap(d4, d10);
    z = Fetch128(s + 144);
    d0 = Add(d0, z);
    d1 = Shuf(kShuf, d1);
    d2 = Add(d2, d0);
    d4 = Xor(d4, z);
    d4 = Xor(d4, d1);
    z = Fetch128(s + 160);
    d5 = Add(d5, z);
    d6 = Shuf(kShuf, d6);
    d8 = Shuf(kShuf, d8);
    d7 = Xor(d7, d5);
    d0 = Xor(d0, z);
    d0 = Xor(d0, d6);
    std::swap(d2, d8);
    z = Fetch128(s + 176);
    d1 = Add(d1, z);
    d2 = Shuf(kShuf, d2);
    d4 = Shuf(kShuf, d4);
    d5 = Mul(kMult, d5);
    d5 = Xor(d5, z);
    d5 = Xor(d5, d2);
    std::swap(d7, d1);
    z = Fetch128(s + 192);
    d6 = Add(d6, z);
    d7 = Shuf(kShuf, d7);
    d0 = Shuf(kShuf, d0);
    d8 = Add(d8, d6);
    d1 = Xor(d1, z);
    d1 = Xor(d1, d7);
    std::swap(d0, d6);
    z = Fetch128(s + 208);
    d2 = Add(d2, z);
    d5 = Shuf(kShuf, d5);
    d4 = Xor(d4, d2);
    d6 = Xor(d6, z);
    d6 = Xor(d6, d9);
    std::swap(d5, d11);
    z = Fetch128(s + 224);
    d7 = Add(d7, z);
    d8 = Shuf(kShuf, d8);
    d1 = Shuf(kShuf, d1);
    d0 = Xor(d0, d7);
    d2 = Xor(d2, z);
    d2 = Xor(d2, d8);
    std::swap(d10, d4);
    z = Fetch128(s + 240);
    d3 = Add(d3, z);
    d4 = Shuf(kShuf, d4);
    d6 = Shuf(kShuf, d6);
    d7 = Mul(kMult, d7);
    d5 = Add(d5, d3);
    d7 = Xor(d7, z);
    d7 = Xor(d7, d4);
    std::swap(d3, d9);
    s += 256;
  } while (s != end);
  d6 = Add(Mul(kMult, d6), _mm_cvtsi64_si128(n));
  if (n % 256 != 0) {
    d7 = Add(_mm_shuffle_epi32(d8, (0 << 6) + (3 << 4) + (2 << 2) + (1 << 0)), d7);
    d8 = Add(Mul(kMult, d8), _mm_cvtsi64_si128(farmhashxo::Hash64(s, n % 256)));
  }
  __m128i t[8];
  d0 = Mul(kMult, Shuf(kShuf, Mul(kMult, d0)));
  d3 = Mul(kMult, Shuf(kShuf, Mul(kMult, d3)));
  d9 = Mul(kMult, Shuf(kShuf, Mul(kMult, d9)));
  d1 = Mul(kMult, Shuf(kShuf, Mul(kMult, d1)));
  d0 = Add(d11, d0);
  d3 = Xor(d7, d3);
  d9 = Add(d8, d9);
  d1 = Add(d10, d1);
  d4 = Add(d3, d4);
  d5 = Add(d9, d5);
  d6 = Xor(d1, d6);
  d2 = Add(d0, d2);
  t[0] = d0;
  t[1] = d3;
  t[2] = d9;
  t[3] = d1;
  t[4] = d4;
  t[5] = d5;
  t[6] = d6;
  t[7] = d2;
  return farmhashxo::Hash64(reinterpret_cast<const char*>(t), sizeof(t));
}

uint64_t Hash64(const char *s, size_t len) {
  // Empirically, farmhashxo seems faster until length 512.
  return len >= 512 ? Hash64Long(s, len, k2, k1) : farmhashxo::Hash64(s, len);
}

uint64_t Hash64WithSeed(const char *s, size_t len, uint64_t seed) {
  return len >= 512 ? Hash64Long(s, len, k1, seed) :
      farmhashxo::Hash64WithSeed(s, len, seed);
}

uint64_t Hash64WithSeeds(const char *s, size_t len, uint64_t seed0, uint64_t seed1) {
  return len >= 512 ? Hash64Long(s, len, seed0, seed1) :
      farmhashxo::Hash64WithSeeds(s, len, seed0, seed1);
}

#endif
}  // namespace farmhashte
