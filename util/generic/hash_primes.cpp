#include "hash_primes.h"
#include "array_size.h"
#include "algorithm.h"

/// Order of fields: reciprocal, reciprocal shift, adjacent hint, divisor
#if defined(_32_)
static constexpr ::NPrivate::THashDivisor PRIME_DIVISORS_HOLDER[]{
    {0x00000000u, 0u, -1, 0xffffffffu}, // guard value, not a valid divisor
    {0x24924925u, 2u, 0, 7u},
    {0xe1e1e1e2u, 4u, 1, 17u},
    {0x1a7b9612u, 4u, 2, 29u},
    {0x3521cfb3u, 5u, 3, 53u},
    {0x51d07eafu, 6u, 4, 97u},
    {0x53909490u, 7u, 5, 193u},
    {0x50f22e12u, 8u, 6, 389u},
    {0x54e3b41au, 9u, 7, 769u},
    {0x53c8eaeeu, 10u, 8, 1543u},
    {0x548eacc6u, 11u, 9, 3079u},
    {0x54f1e41eu, 12u, 10, 6151u},
    {0x554e390au, 13u, 11, 12289u},
    {0x5518ee41u, 14u, 12, 24593u},
    {0x554c7203u, 15u, 13, 49157u},
    {0x5549c781u, 16u, 14, 98317u},
    {0x55531c76u, 17u, 15, 196613u},
    {0x554fc734u, 18u, 16, 393241u},
    {0x555538e4u, 19u, 17, 786433u},
    {0x55550e39u, 20u, 18, 1572869u},
    {0x5555071du, 21u, 19, 3145739u},
    {0x5555271du, 22u, 20, 6291469u},
    {0x55554c72u, 23u, 21, 12582917u},
    {0x55554472u, 24u, 22, 25165843u},
    {0x5555531du, 25u, 23, 50331653u},
    {0x55555039u, 26u, 24, 100663319u},
    {0x55555339u, 27u, 25, 201326611u},
    {0x5555550fu, 28u, 26, 402653189u},
    {0x555552ddu, 29u, 27, 805306457u},
    {0x55555544u, 30u, 28, 1610612741u},
    {0x55555554u, 31u, 29, 3221225473u},
    {0x00000006u, 31u, 30, 4294967291u},
};
#else
static constexpr ::NPrivate::THashDivisor PRIME_DIVISORS_HOLDER[]{
    {0x0000000000000000ul, 0u, -1, 0xffffffffu}, // guard value, not a valid divisor
    {0x2492492492492493ul, 2u, 0, 7u},
    {0xe1e1e1e1e1e1e1e2ul, 4u, 1, 17u},
    {0x1a7b9611a7b9611bul, 4u, 2, 29u},
    {0x3521cfb2b78c1353ul, 5u, 3, 53u},
    {0x51d07eae2f8151d1ul, 6u, 4, 97u},
    {0x5390948f40feac70ul, 7u, 5, 193u},
    {0x50f22e111c4c56dful, 8u, 6, 389u},
    {0x54e3b4194ce65de1ul, 9u, 7, 769u},
    {0x53c8eaedea6e7f17ul, 10u, 8, 1543u},
    {0x548eacc5e1e6e3fcul, 11u, 9, 3079u},
    {0x54f1e41d7767d70cul, 12u, 10, 6151u},
    {0x554e39097a781d80ul, 13u, 11, 12289u},
    {0x5518ee4079ea6929ul, 14u, 12, 24593u},
    {0x554c72025d459231ul, 15u, 13, 49157u},
    {0x5549c78094504ff3ul, 16u, 14, 98317u},
    {0x55531c757b3c329cul, 17u, 15, 196613u},
    {0x554fc7339753b424ul, 18u, 16, 393241u},
    {0x555538e39097b3f4ul, 19u, 17, 786433u},
    {0x55550e38f25ecd82ul, 20u, 18, 1572869u},
    {0x5555071c83b421d2ul, 21u, 19, 3145739u},
    {0x5555271c78097a6aul, 22u, 20, 6291469u},
    {0x55554c71c757b425ul, 23u, 21, 12582917u},
    {0x55554471c7f25ec7ul, 24u, 22, 25165843u},
    {0x5555531c71cad098ul, 25u, 23, 50331653u},
    {0x55555038e3a1d098ul, 26u, 24, 100663319u},
    {0x55555338e3919098ul, 27u, 25, 201326611u},
    {0x5555550e38e39d0aul, 28u, 26, 402653189u},
    {0x555552dc71cbb1eeul, 29u, 27, 805306457u},
    {0x555555438e38e47cul, 30u, 28, 1610612741u},
    {0x555555538e38e391ul, 31u, 29, 3221225473u},
    {0x000000050000001aul, 31u, 30, 4294967291u},
};
#endif

static constexpr const ::NPrivate::THashDivisor* PRIME_DIVISORS = &PRIME_DIVISORS_HOLDER[1]; ///< Address of the first valid divisor
static constexpr size_t PRIME_DIVISORS_SIZE = Y_ARRAY_SIZE(PRIME_DIVISORS_HOLDER) - 1;       ///< Number of valid divisors without the guarding value

unsigned long HashBucketCount(unsigned long elementCount) {
    return HashBucketCountExt(elementCount)();
}

static inline ::NPrivate::THashDivisor HashBucketBoundedSearch(unsigned long elementCount) {
    const auto begin = PRIME_DIVISORS;
    const auto end = PRIME_DIVISORS + PRIME_DIVISORS_SIZE - 1; // adjust range so the last element will be returned if elementCount is bigger than all PRIME_DIVISORS
    return *LowerBoundBy(begin, end, elementCount, std::mem_fn(&::NPrivate::THashDivisor::Divisor));
}

Y_CONST_FUNCTION
::NPrivate::THashDivisor HashBucketCountExt(unsigned long elementCount) {
    if (elementCount <= PRIME_DIVISORS[0]()) {
        return PRIME_DIVISORS[0];
    }

    return HashBucketBoundedSearch(elementCount);
}

Y_CONST_FUNCTION
::NPrivate::THashDivisor HashBucketCountExt(unsigned long elementCount, int hint) {
    if (Y_LIKELY(static_cast<size_t>(hint) < PRIME_DIVISORS_SIZE)) {
        const int index = hint;
        const ::NPrivate::THashDivisor* cnd = PRIME_DIVISORS + index;
        if (Y_LIKELY(elementCount <= cnd->Divisor)) {
            const ::NPrivate::THashDivisor* prev = cnd - 1;
            static_assert(~PRIME_DIVISORS[-1].Divisor == 0, "Invalid guard");
            /*
            If index == 0 then PRIME_DIVISORS[0] should be returned.
            Otherwise `cnd` is correct value iff (prev->Divisor < elementCount).
            Ergo hint is correct if (index == 0 || prev->Divisor < elementCount);
            But we can set guard's value to -1 and check both conditions at once.
            */
            if (Y_LIKELY(prev->Divisor + 1u <= elementCount)) {
                return *cnd;
            }
        }
    }

    return HashBucketBoundedSearch(elementCount);
}
