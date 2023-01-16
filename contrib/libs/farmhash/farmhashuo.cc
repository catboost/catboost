#include "common.h"

namespace {
    #include "farmhashna.cc"
}

namespace farmhashuo {
#undef Fetch
#define Fetch Fetch64

#undef Rotate
#define Rotate Rotate64

STATIC_INLINE uint64_t H(uint64_t x, uint64_t y, uint64_t mul, int r) {
  uint64_t a = (x ^ y) * mul;
  a ^= (a >> 47);
  uint64_t b = (y ^ a) * mul;
  return Rotate(b, r) * mul;
}

uint64_t Hash64WithSeeds(const char *s, size_t len,
                         uint64_t seed0, uint64_t seed1) {
  if (len <= 64) {
    return farmhashna::Hash64WithSeeds(s, len, seed0, seed1);
  }

  // For strings over 64 bytes we loop.  Internal state consists of
  // 64 bytes: u, v, w, x, y, and z.
  uint64_t x = seed0;
  uint64_t y = seed1 * k2 + 113;
  uint64_t z = farmhashna::ShiftMix(y * k2) * k2;
  pair<uint64_t, uint64_t> v = make_pair(seed0, seed1);
  pair<uint64_t, uint64_t> w = make_pair(0, 0);
  uint64_t u = x - z;
  x *= k2;
  uint64_t mul = k2 + (u & 0x82);

  // Set end so that after the loop we have 1 to 64 bytes left to process.
  const char* end = s + ((len - 1) / 64) * 64;
  const char* last64 = end + ((len - 1) & 63) - 63;
  assert(s + len - 64 == last64);
  do {
    uint64_t a0 = Fetch(s);
    uint64_t a1 = Fetch(s + 8);
    uint64_t a2 = Fetch(s + 16);
    uint64_t a3 = Fetch(s + 24);
    uint64_t a4 = Fetch(s + 32);
    uint64_t a5 = Fetch(s + 40);
    uint64_t a6 = Fetch(s + 48);
    uint64_t a7 = Fetch(s + 56);
    x += a0 + a1;
    y += a2;
    z += a3;
    v.first += a4;
    v.second += a5 + a1;
    w.first += a6;
    w.second += a7;

    x = Rotate(x, 26);
    x *= 9;
    y = Rotate(y, 29);
    z *= mul;
    v.first = Rotate(v.first, 33);
    v.second = Rotate(v.second, 30);
    w.first ^= x;
    w.first *= 9;
    z = Rotate(z, 32);
    z += w.second;
    w.second += z;
    z *= 9;
    std::swap(u, y);

    z += a0 + a6;
    v.first += a2;
    v.second += a3;
    w.first += a4;
    w.second += a5 + a6;
    x += a1;
    y += a7;

    y += v.first;
    v.first += x - y;
    v.second += w.first;
    w.first += v.second;
    w.second += x - y;
    x += w.second;
    w.second = Rotate(w.second, 34);
    std::swap(u, z);
    s += 64;
  } while (s != end);
  // Make s point to the last 64 bytes of input.
  s = last64;
  u *= 9;
  v.second = Rotate(v.second, 28);
  v.first = Rotate(v.first, 20);
  w.first += ((len - 1) & 63);
  u += y;
  y += u;
  x = Rotate(y - x + v.first + Fetch(s + 8), 37) * mul;
  y = Rotate(y ^ v.second ^ Fetch(s + 48), 42) * mul;
  x ^= w.second * 9;
  y += v.first + Fetch(s + 40);
  z = Rotate(z + w.first, 33) * mul;
  v = farmhashna::WeakHashLen32WithSeeds(s, v.second * mul, x + w.first);
  w = farmhashna::WeakHashLen32WithSeeds(s + 32, z + w.second, y + Fetch(s + 16));
  return H(farmhashna::HashLen16(v.first + x, w.first ^ y, mul) + z - u,
           H(v.second + y, w.second + z, k2, 30) ^ x,
           k2,
           31);
}

uint64_t Hash64WithSeed(const char *s, size_t len, uint64_t seed) {
  return len <= 64 ? farmhashna::Hash64WithSeed(s, len, seed) :
      Hash64WithSeeds(s, len, 0, seed);
}

uint64_t Hash64(const char *s, size_t len) {
  return len <= 64 ? farmhashna::Hash64(s, len) :
      Hash64WithSeeds(s, len, 81, 0);
}
}  // namespace farmhashuo
