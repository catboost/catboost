#include "common.h"

#include "farmhash_iface.h"

#include <util/system/cpu_id.h>

namespace NAMESPACE_FOR_HASH_FUNCTIONS {

// BASIC STRING HASHING

// Hash function for a byte array.  See also Hash(), below.
// May change from time to time, may differ on different platforms, may differ
// depending on NDEBUG.
uint32_t Hash32(const char* s, size_t len) {
  return DebugTweak(
      (NX86::CachedHaveSSE41() & x86_64) ? farmhashnt::Hash32(s, len) :
      (NX86::CachedHaveSSE42() & NX86::CachedHaveAES()) ? farmhashsu::Hash32(s, len) :
      NX86::CachedHaveSSE42() ? farmhashsa::Hash32(s, len) :
      farmhashmk::Hash32(s, len));
}

// Hash function for a byte array.  For convenience, a 32-bit seed is also
// hashed into the result.
// May change from time to time, may differ on different platforms, may differ
// depending on NDEBUG.
uint32_t Hash32WithSeed(const char* s, size_t len, uint32_t seed) {
  return DebugTweak(
      (NX86::CachedHaveSSE41() & x86_64) ? farmhashnt::Hash32WithSeed(s, len, seed) :
      (NX86::CachedHaveSSE42() & NX86::CachedHaveAES()) ? farmhashsu::Hash32WithSeed(s, len, seed) :
      NX86::CachedHaveSSE42() ? farmhashsa::Hash32WithSeed(s, len, seed) :
      farmhashmk::Hash32WithSeed(s, len, seed));
}

// Hash function for a byte array.  For convenience, a 64-bit seed is also
// hashed into the result.  See also Hash(), below.
// May change from time to time, may differ on different platforms, may differ
// depending on NDEBUG.
uint64_t Hash64(const char* s, size_t len) {
  return DebugTweak(
      (NX86::CachedHaveSSE42() & x86_64) ?
      farmhashte::Hash64(s, len) :
      farmhashxo::Hash64(s, len));
}

// Hash function for a byte array.
// May change from time to time, may differ on different platforms, may differ
// depending on NDEBUG.
size_t Hash(const char* s, size_t len) {
  return sizeof(size_t) == 8 ? Hash64(s, len) : Hash32(s, len);
}

// Hash function for a byte array.  For convenience, a 64-bit seed is also
// hashed into the result.
// May change from time to time, may differ on different platforms, may differ
// depending on NDEBUG.
uint64_t Hash64WithSeed(const char* s, size_t len, uint64_t seed) {
  return DebugTweak(farmhashna::Hash64WithSeed(s, len, seed));
}

// Hash function for a byte array.  For convenience, two seeds are also
// hashed into the result.
// May change from time to time, may differ on different platforms, may differ
// depending on NDEBUG.
uint64_t Hash64WithSeeds(const char* s, size_t len, uint64_t seed0, uint64_t seed1) {
  return DebugTweak(farmhashna::Hash64WithSeeds(s, len, seed0, seed1));
}

// Hash function for a byte array.
// May change from time to time, may differ on different platforms, may differ
// depending on NDEBUG.
uint128_t Hash128(const char* s, size_t len) {
  return DebugTweak(farmhashcc::Fingerprint128(s, len));
}

// Hash function for a byte array.  For convenience, a 128-bit seed is also
// hashed into the result.
// May change from time to time, may differ on different platforms, may differ
// depending on NDEBUG.
uint128_t Hash128WithSeed(const char* s, size_t len, uint128_t seed) {
  return DebugTweak(farmhashcc::CityHash128WithSeed(s, len, seed));
}

// BASIC NON-STRING HASHING

// FINGERPRINTING (i.e., good, portable, forever-fixed hash functions)

// Fingerprint function for a byte array.  Most useful in 32-bit binaries.
uint32_t Fingerprint32(const char* s, size_t len) {
  return farmhashmk::Hash32(s, len);
}

// Fingerprint function for a byte array.
uint64_t Fingerprint64(const char* s, size_t len) {
  return farmhashna::Hash64(s, len);
}

// Fingerprint function for a byte array.
uint128_t Fingerprint128(const char* s, size_t len) {
  return farmhashcc::Fingerprint128(s, len);
}

// Older and still available but perhaps not as fast as the above:
//   farmhashns::Hash32{,WithSeed}()

}  // namespace NAMESPACE_FOR_HASH_FUNCTIONS

