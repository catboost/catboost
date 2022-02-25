#include "common.h"

namespace {
    #include "farmhashsu.cc"
}

namespace farmhashsa {
#if !can_use_sse42

uint32_t Hash32(const char *s, size_t len) {
  FARMHASH_DIE_IF_MISCONFIGURED;
  return s == NULL ? 0 : len;
}

uint32_t Hash32WithSeed(const char *s, size_t len, uint32_t seed) {
  FARMHASH_DIE_IF_MISCONFIGURED;
  return seed + Hash32(s, len);
}

#else

#undef Fetch
#define Fetch Fetch32

#undef Rotate
#define Rotate Rotate32

#undef Bswap
#define Bswap Bswap32

// Helpers for data-parallel operations (4x 32-bits).
STATIC_INLINE __m128i Add(__m128i x, __m128i y) { return _mm_add_epi32(x, y); }
STATIC_INLINE __m128i Xor(__m128i x, __m128i y) { return _mm_xor_si128(x, y); }
STATIC_INLINE __m128i Or(__m128i x, __m128i y) { return _mm_or_si128(x, y); }
STATIC_INLINE __m128i Mul(__m128i x, __m128i y) { return _mm_mullo_epi32(x, y); }
STATIC_INLINE __m128i Mul5(__m128i x) { return Add(x, _mm_slli_epi32(x, 2)); }
STATIC_INLINE __m128i Rotate(__m128i x, int c) {
  return Or(_mm_slli_epi32(x, c),
            _mm_srli_epi32(x, 32 - c));
}
STATIC_INLINE __m128i Rot17(__m128i x) { return Rotate(x, 17); }
STATIC_INLINE __m128i Rot19(__m128i x) { return Rotate(x, 19); }
STATIC_INLINE __m128i Shuffle0321(__m128i x) {
  return _mm_shuffle_epi32(x, (0 << 6) + (3 << 4) + (2 << 2) + (1 << 0));
}

uint32_t Hash32(const char *s, size_t len) {
  const uint32_t seed = 81;
  if (len <= 24) {
    return len <= 12 ?
        (len <= 4 ?
         farmhashmk::Hash32Len0to4(s, len) :
         farmhashmk::Hash32Len5to12(s, len)) :
        farmhashmk::Hash32Len13to24(s, len);
  }

  if (len < 40) {
    uint32_t a = len, b = seed * c2, c = a + b;
    a += Fetch(s + len - 4);
    b += Fetch(s + len - 20);
    c += Fetch(s + len - 16);
    uint32_t d = a;
    a = NAMESPACE_FOR_HASH_FUNCTIONS::Rotate32(a, 21);
    a = Mur(a, Mur(b, Mur(c, d)));
    a += Fetch(s + len - 12);
    b += Fetch(s + len - 8);
    d += a;
    a += d;
    b = Mur(b, d) * c2;
    a = _mm_crc32_u32(a, b + c);
    return farmhashmk::Hash32Len13to24(s, (len + 1) / 2, a) + b;
  }

#undef Mulc1
#define Mulc1(x) Mul((x), cc1)

#undef Mulc2
#define Mulc2(x) Mul((x), cc2)

#undef Murk
#define Murk(a, h)                              \
  Add(k,                                        \
      Mul5(                                     \
          Rot19(                                \
              Xor(                              \
                  Mulc2(                        \
                      Rot17(                    \
                          Mulc1(a))),           \
                  (h)))))

  const __m128i cc1 = _mm_set1_epi32(c1);
  const __m128i cc2 = _mm_set1_epi32(c2);
  __m128i h = _mm_set1_epi32(seed);
  __m128i g = _mm_set1_epi32(c1 * seed);
  __m128i f = g;
  __m128i k = _mm_set1_epi32(0xe6546b64);
  if (len < 80) {
    __m128i a = Fetch128(s);
    __m128i b = Fetch128(s + 16);
    __m128i c = Fetch128(s + (len - 15) / 2);
    __m128i d = Fetch128(s + len - 32);
    __m128i e = Fetch128(s + len - 16);
    h = Add(h, a);
    g = Add(g, b);
    g = Shuffle0321(g);
    f = Add(f, c);
    __m128i be = Add(b, Mulc1(e));
    h = Add(h, f);
    f = Add(f, h);
    h = Add(Murk(d, h), e);
    k = Xor(k, _mm_shuffle_epi8(g, f));
    g = Add(Xor(c, g), a);
    f = Add(Xor(be, f), d);
    k = Add(k, be);
    k = Add(k, _mm_shuffle_epi8(f, h));
    f = Add(f, g);
    g = Add(g, f);
    g = Add(_mm_set1_epi32(len), Mulc1(g));
  } else {
    // len >= 80
    // The following is loosely modelled after farmhashmk::Hash32.
    size_t iters = (len - 1) / 80;
    len -= iters * 80;

#undef Chunk
#define Chunk() do {                            \
  __m128i a = Fetch128(s);                       \
  __m128i b = Fetch128(s + 16);                  \
  __m128i c = Fetch128(s + 32);                  \
  __m128i d = Fetch128(s + 48);                  \
  __m128i e = Fetch128(s + 64);                  \
  h = Add(h, a);                                \
  g = Add(g, b);                                \
  g = Shuffle0321(g);                           \
  f = Add(f, c);                                \
  __m128i be = Add(b, Mulc1(e));                \
  h = Add(h, f);                                \
  f = Add(f, h);                                \
  h = Add(Murk(d, h), e);                       \
  k = Xor(k, _mm_shuffle_epi8(g, f));           \
  g = Add(Xor(c, g), a);                        \
  f = Add(Xor(be, f), d);                       \
  k = Add(k, be);                               \
  k = Add(k, _mm_shuffle_epi8(f, h));           \
  f = Add(f, g);                                \
  g = Add(g, f);                                \
  f = Mulc1(f);                                 \
} while (0)

    while (iters-- != 0) {
      Chunk();
      s += 80;
    }

    if (len != 0) {
      h = Add(h, _mm_set1_epi32(len));
      s = s + len - 80;
      Chunk();
    }
  }

  g = Shuffle0321(g);
  k = Xor(k, g);
  f = Mulc1(f);
  k = Mulc2(k);
  g = Mulc1(g);
  h = Mulc2(h);
  k = Add(k, _mm_shuffle_epi8(g, f));
  h = Add(h, f);
  f = Add(f, h);
  g = Add(g, k);
  k = Add(k, g);
  k = Xor(k, _mm_shuffle_epi8(f, h));
  __m128i buf[4];
  buf[0] = f;
  buf[1] = g;
  buf[2] = k;
  buf[3] = h;
  s = reinterpret_cast<char*>(buf);
  uint32_t x = Fetch(s);
  uint32_t y = Fetch(s+4);
  uint32_t z = Fetch(s+8);
  x = _mm_crc32_u32(x, Fetch(s+12));
  y = _mm_crc32_u32(y, Fetch(s+16));
  z = _mm_crc32_u32(z * c1, Fetch(s+20));
  x = _mm_crc32_u32(x, Fetch(s+24));
  y = _mm_crc32_u32(y * c1, Fetch(s+28));
  uint32_t o = y;
  z = _mm_crc32_u32(z, Fetch(s+32));
  x = _mm_crc32_u32(x * c1, Fetch(s+36));
  y = _mm_crc32_u32(y, Fetch(s+40));
  z = _mm_crc32_u32(z * c1, Fetch(s+44));
  x = _mm_crc32_u32(x, Fetch(s+48));
  y = _mm_crc32_u32(y * c1, Fetch(s+52));
  z = _mm_crc32_u32(z, Fetch(s+56));
  x = _mm_crc32_u32(x, Fetch(s+60));
  return (o - x + y - z) * c1;
}

#undef Chunk
#undef Murk
#undef Mulc2
#undef Mulc1

uint32_t Hash32WithSeed(const char *s, size_t len, uint32_t seed) {
  if (len <= 24) {
    if (len >= 13) return farmhashmk::Hash32Len13to24(s, len, seed * c1);
    else if (len >= 5) return farmhashmk::Hash32Len5to12(s, len, seed);
    else return farmhashmk::Hash32Len0to4(s, len, seed);
  }
  uint32_t h = farmhashmk::Hash32Len13to24(s, 24, seed ^ len);
  return _mm_crc32_u32(Hash32(s + 24, len - 24) + seed, h);
}

#endif
}  // namespace farmhashsa
