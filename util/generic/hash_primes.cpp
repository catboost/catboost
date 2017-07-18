#include "hash_primes.h"
#include "array_size.h"
#include "algorithm.h"

static const unsigned long _y_prime_list[] = {
    7ul, 17ul, 29ul,
    53ul, 97ul, 193ul, 389ul, 769ul,
    1543ul, 3079ul, 6151ul, 12289ul, 24593ul,
    49157ul, 98317ul, 196613ul, 393241ul, 786433ul,
    1572869ul, 3145739ul, 6291469ul, 12582917ul, 25165843ul,
    50331653ul, 100663319ul, 201326611ul, 402653189ul, 805306457ul,
    1610612741ul, 3221225473ul, 4294967291ul,
};

const unsigned long* const _y_first_prime = _y_prime_list;
const unsigned long* const _y_last_prime = _y_prime_list + Y_ARRAY_SIZE(_y_prime_list) - 1;

unsigned long HashBucketCount(unsigned long elementCount) {
    if (elementCount <= *_y_first_prime) {
        return *_y_first_prime;
    }

    return *LowerBound(_y_first_prime, _y_last_prime, elementCount);
}
