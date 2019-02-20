#include "hash_primes.h"
#include "array_size.h"
#include "algorithm.h"

static const unsigned long _y_prime_list[] = {
    7ul,
    17ul,
    29ul,
    53ul,
    97ul,
    193ul,
    389ul,
    769ul,
    1543ul,
    3079ul,
    6151ul,
    12289ul,
    24593ul,
    49157ul,
    98317ul,
    196613ul,
    393241ul,
    786433ul,
    1572869ul,
    3145739ul,
    6291469ul,
    12582917ul,
    25165843ul,
    50331653ul,
    100663319ul,
    201326611ul,
    402653189ul,
    805306457ul,
    1610612741ul,
    3221225473ul,
    4294967291ul,
};

#if defined(_32_)
static constexpr ::NPrivate::THashDivisor PRIME_DIVISORS[]{
    {0x24924925u,  2u, 7u},
    {0xe1e1e1e2u,  4u, 17u},
    {0x1a7b9612u,  4u, 29u},
    {0x3521cfb3u,  5u, 53u},
    {0x51d07eafu,  6u, 97u},
    {0x53909490u,  7u, 193u},
    {0x50f22e12u,  8u, 389u},
    {0x54e3b41au,  9u, 769u},
    {0x53c8eaeeu, 10u, 1543u},
    {0x548eacc6u, 11u, 3079u},
    {0x54f1e41eu, 12u, 6151u},
    {0x554e390au, 13u, 12289u},
    {0x5518ee41u, 14u, 24593u},
    {0x554c7203u, 15u, 49157u},
    {0x5549c781u, 16u, 98317u},
    {0x55531c76u, 17u, 196613u},
    {0x554fc734u, 18u, 393241u},
    {0x555538e4u, 19u, 786433u},
    {0x55550e39u, 20u, 1572869u},
    {0x5555071du, 21u, 3145739u},
    {0x5555271du, 22u, 6291469u},
    {0x55554c72u, 23u, 12582917u},
    {0x55554472u, 24u, 25165843u},
    {0x5555531du, 25u, 50331653u},
    {0x55555039u, 26u, 100663319u},
    {0x55555339u, 27u, 201326611u},
    {0x5555550fu, 28u, 402653189u},
    {0x555552ddu, 29u, 805306457u},
    {0x55555544u, 30u, 1610612741u},
    {0x55555554u, 31u, 3221225473u},
    {0x00000006u, 31u, 4294967291u},
};
#else
static constexpr ::NPrivate::THashDivisor PRIME_DIVISORS[]{
    {0x2492492492492493ull,  2u, 7u},
    {0xe1e1e1e1e1e1e1e2ull,  4u, 17u},
    {0x1a7b9611a7b9611bull,  4u, 29u},
    {0x3521cfb2b78c1353ull,  5u, 53u},
    {0x51d07eae2f8151d1ull,  6u, 97u},
    {0x5390948f40feac70ull,  7u, 193u},
    {0x50f22e111c4c56dfull,  8u, 389u},
    {0x54e3b4194ce65de1ull,  9u, 769u},
    {0x53c8eaedea6e7f17ull, 10u, 1543u},
    {0x548eacc5e1e6e3fcull, 11u, 3079u},
    {0x54f1e41d7767d70cull, 12u, 6151u},
    {0x554e39097a781d80ull, 13u, 12289u},
    {0x5518ee4079ea6929ull, 14u, 24593u},
    {0x554c72025d459231ull, 15u, 49157u},
    {0x5549c78094504ff3ull, 16u, 98317u},
    {0x55531c757b3c329cull, 17u, 196613u},
    {0x554fc7339753b424ull, 18u, 393241u},
    {0x555538e39097b3f4ull, 19u, 786433u},
    {0x55550e38f25ecd82ull, 20u, 1572869u},
    {0x5555071c83b421d2ull, 21u, 3145739u},
    {0x5555271c78097a6aull, 22u, 6291469u},
    {0x55554c71c757b425ull, 23u, 12582917u},
    {0x55554471c7f25ec7ull, 24u, 25165843u},
    {0x5555531c71cad098ull, 25u, 50331653u},
    {0x55555038e3a1d098ull, 26u, 100663319u},
    {0x55555338e3919098ull, 27u, 201326611u},
    {0x5555550e38e39d0aull, 28u, 402653189u},
    {0x555552dc71cbb1eeull, 29u, 805306457u},
    {0x555555438e38e47cull, 30u, 1610612741u},
    {0x555555538e38e391ull, 31u, 3221225473u},
    {0x000000050000001aull, 31u, 4294967291u},
};
#endif

const unsigned long* const _y_first_prime = _y_prime_list;
const unsigned long* const _y_last_prime = _y_prime_list + Y_ARRAY_SIZE(_y_prime_list) - 1;

unsigned long HashBucketCount(unsigned long elementCount) {
    return HashBucketCountExt(elementCount)();
}


Y_CONST_FUNCTION
const ::NPrivate::THashDivisor& HashBucketCountExt(unsigned long elementCount) {
    if (elementCount <= PRIME_DIVISORS[0]()) {
        return PRIME_DIVISORS[0];
    }

    return *LowerBoundBy(std::begin(PRIME_DIVISORS), std::end(PRIME_DIVISORS) - 1, elementCount, std::mem_fn(&::NPrivate::THashDivisor::Divisor));
}

