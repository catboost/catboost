#pragma once
#ifndef YT_ALLOC_INL_H_
#error "Direct inclusion of this file is not allowed, include ytalloc.h"
// For the sake of sane code completion.
#include "ytalloc.h"
#endif

#include <util/system/types.h>

namespace NYT::NYTAlloc {

////////////////////////////////////////////////////////////////////////////////

// Maps small chunk ranks to size in bytes.
constexpr ui16 SmallRankToSize[SmallRankCount] = {
    0,
    16, 32, 48, 64, 96, 128,
    192, 256, 384, 512, 768, 1024, 1536, 2048,
    3072, 4096, 6144, 8192, 12288, 16384, 24576, 32768
};

// Helper array for mapping size to small chunk rank.
constexpr ui8 SizeToSmallRank1[65] = {
    1, 1, 1, 2, 2, // 16, 32
    3, 3, 4, 4, // 48, 64
    5, 5, 5, 5, 6, 6, 6, 6, // 96, 128
    7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, // 192, 256
    9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, // 384
    10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10  // 512
};

// Helper array for mapping size to small chunk rank.
constexpr unsigned char SizeToSmallRank2[128] = {
    10, 10, 11, 12, // 512, 512, 768, 1022
    13, 13, 14, 14, // 1536, 2048
    15, 15, 15, 15, 16, 16, 16, 16, // 3072, 4096
    17, 17, 17, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 18, 18, // 6144, 8192
    19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, // 12288
    20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, // 16384
    21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
    21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, // 22576
    22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22,
    22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22  // 32768
};

////////////////////////////////////////////////////////////////////////////////

constexpr size_t SizeToSmallRank(size_t size)
{
    if (size <= 512) {
        return SizeToSmallRank1[(size + 7) >> 3];
    } else {
        if (size <= LargeAllocationSizeThreshold) {
            return SizeToSmallRank2[(size - 1) >> 8];
        } else {
            return 0;
        }
    }
}

void* AllocateSmall(size_t rank);

template <size_t Size>
void* AllocateConstSize()
{
    constexpr auto rank = SizeToSmallRank(Size);
    if constexpr(rank != 0) {
        return AllocateSmall(rank);
    } else {
        return Allocate(Size);
    }
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NYTAlloc
