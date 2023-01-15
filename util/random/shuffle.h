#pragma once

#include "fast.h"
#include "entropy.h"

#include <util/generic/utility.h>

// some kind of https://en.wikipedia.org/wiki/Fisher–Yates_shuffle#The_modern_algorithm

template <typename TRandIter, typename TRandIterEnd>
inline void Shuffle(TRandIter begin, TRandIterEnd end) {
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
    const size_t sz = end - begin;

    for (size_t i = 1; i < sz; ++i) {
        DoSwap(*(begin + i), *(begin + gen.Uniform(i + 1)));
    }
}

template <typename TRange>
inline void ShuffleRange(TRange& range) {
    Shuffle(range.begin(), range.end());
}

template <typename TRange, typename TRandGen>
inline void ShuffleRange(TRange& range, TRandGen&& gen) {
    Shuffle(range.begin(), range.end(), gen);
}
