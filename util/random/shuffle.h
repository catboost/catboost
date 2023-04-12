#pragma once

#include "fast.h"
#include "entropy.h"

#include <util/generic/utility.h>
#include <util/system/yassert.h>

// some kind of https://en.wikipedia.org/wiki/Fisher–Yates_shuffle#The_modern_algorithm

template <typename TRandIter, typename TRandIterEnd>
inline void Shuffle(TRandIter begin, TRandIterEnd end) {
    Y_ASSERT(begin <= end);
    static_assert(sizeof(end - begin) <= sizeof(size_t), "fixme");
    static_assert(sizeof(TReallyFastRng32::RandMax()) <= sizeof(size_t), "fixme");

    if ((size_t)(end - begin) < (size_t)TReallyFastRng32::RandMax()) {
        Shuffle(begin, end, TReallyFastRng32(Seed()));
    } else {
        Shuffle(begin, end, TFastRng64(Seed()));
    }
}

template <typename TRandIter, typename TRandIterEnd, typename TRandGen>
inline void Shuffle(TRandIter begin, TRandIterEnd end, TRandGen&& gen) {
    Y_ASSERT(begin <= end);
    const size_t sz = end - begin;

    for (size_t i = 1; i < sz; ++i) {
        DoSwap(*(begin + i), *(begin + gen.Uniform(i + 1)));
    }
}

// Fills first size elements of array with equiprobably randomly
// chosen elements of array with no replacement
template <typename TRandIter, typename TRandIterEnd>
inline void PartialShuffle(TRandIter begin, TRandIterEnd end, size_t size) {
    Y_ASSERT(begin <= end);
    static_assert(sizeof(end - begin) <= sizeof(size_t), "fixme");
    static_assert(sizeof(TReallyFastRng32::RandMax()) <= sizeof(size_t), "fixme");

    if ((size_t)(end - begin) < (size_t)TReallyFastRng32::RandMax()) {
        PartialShuffle(begin, end, size, TReallyFastRng32(Seed()));
    } else {
        PartialShuffle(begin, end, size, TFastRng64(Seed()));
    }
}

template <typename TRandIter, typename TRandIterEnd, typename TRandGen>
inline void PartialShuffle(TRandIter begin, TRandIterEnd end, size_t size, TRandGen&& gen) {
    Y_ASSERT(begin <= end);

    const size_t totalSize = end - begin;
    Y_ASSERT(size <= totalSize); // Size of shuffled part should be less than or equal to the size of container
    if (totalSize == 0) {
        return;
    }
    size = Min(size, totalSize - 1);

    for (size_t i = 0; i < size; ++i) {
        DoSwap(*(begin + i), *(begin + gen.Uniform(i, totalSize)));
    }
}

template <typename TRange>
inline void ShuffleRange(TRange& range) {
    auto b = range.begin();

    Shuffle(b, range.end());
}

template <typename TRange, typename TRandGen>
inline void ShuffleRange(TRange& range, TRandGen&& gen) {
    auto b = range.begin();

    Shuffle(b, range.end(), gen);
}
