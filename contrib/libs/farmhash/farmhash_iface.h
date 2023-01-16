#define FARMHASH_INTERFACE(name) namespace name {                                        \
    uint32_t Hash32(const char* s, size_t len);                                          \
    uint32_t Hash32WithSeed(const char* s, size_t len, uint32_t seed);                   \
    uint64_t Hash64(const char* s, size_t len);                                          \
    uint64_t Hash64WithSeed(const char* s, size_t len, uint64_t seed);                   \
    uint64_t Hash64WithSeeds(const char* s, size_t len, uint64_t seed0, uint64_t seed1); \
    uint128_t Hash128(const char* s, size_t len);                                        \
    uint128_t Hash128WithSeed(const char* s, size_t len, uint128_t seed);                \
}

namespace farmhashcc {
    uint32_t Fingerprint32(const char* s, size_t len);
    uint64_t Fingerprint64(const char* s, size_t len);
    uint128_t Fingerprint128(const char* s, size_t len);
    uint128_t CityHash128WithSeed(const char* s, size_t len, uint128_t seed);
}

FARMHASH_INTERFACE(farmhashcc)
FARMHASH_INTERFACE(farmhashsa)
FARMHASH_INTERFACE(farmhashsu)
FARMHASH_INTERFACE(farmhashmk)
FARMHASH_INTERFACE(farmhashnt)
FARMHASH_INTERFACE(farmhashte)
FARMHASH_INTERFACE(farmhashxo)
FARMHASH_INTERFACE(farmhashuo)
FARMHASH_INTERFACE(farmhashna)

#undef FARMHASH_INTERFACE
