#include "murmur.h"

#include <util/system/unaligned_mem.h>

namespace NMurmurPrivate {
    //-----------------------------------------------------------------------------
    // MurmurHash2, by Austin Appleby

    // Note - This code makes a few assumptions about how your machine behaves -

    // 1. We can read a 4-byte value from any address without crashing
    // 2. sizeof(int) == 4

    // And it has a few limitations -

    // 1. It will not work incrementally.
    // 2. It will not produce the same results on little-endian and big-endian
    //    machines.

    Y_NO_INLINE ui32 MurmurHash32(const void* key, size_t len, ui32 seed) noexcept {
        const ui32 m = 0x5bd1e995;
        const int r = 24;
        ui32 h = ui32(seed ^ len);

        TUnalignedMemoryIterator<ui32> iter(key, len);

        while (!iter.AtEnd()) {
            ui32 k = iter.Next();

            k *= m;
            k ^= k >> r;
            k *= m;

            h *= m;
            h ^= k;
        }

        const unsigned char* data = iter.Last();

        switch (iter.Left()) {
            case 3:
                h ^= data[2] << 16;
                [[fallthrough]];

            case 2:
                h ^= data[1] << 8;
                [[fallthrough]];

            case 1:
                h ^= data[0];
                h *= m;
                break;
        }

        h ^= h >> 13;
        h *= m;
        h ^= h >> 15;

        return h;
    }

    //-----------------------------------------------------------------------------
    // MurmurHash2, 64-bit versions, by Austin Appleby

    // The same caveats as 32-bit MurmurHash2 apply here - beware of alignment
    // and endian-ness issues if used across multiple platforms.

    // 64-bit hash for 64-bit platforms

    Y_NO_INLINE ui64 MurmurHash64(const void* key, size_t len, ui64 seed) noexcept {
        const ui64 m = ULL(0xc6a4a7935bd1e995);
        const int r = 47;

        ui64 h = seed ^ (len * m);
        TUnalignedMemoryIterator<ui64> iter(key, len);

        while (!iter.AtEnd()) {
            ui64 k = iter.Next();

            k *= m;
            k ^= k >> r;
            k *= m;

            h ^= k;
            h *= m;
        }

        const unsigned char* data2 = iter.Last();

        switch (iter.Left()) {
            case 7:
                h ^= ui64(data2[6]) << 48;
                [[fallthrough]];

            case 6:
                h ^= ui64(data2[5]) << 40;
                [[fallthrough]];

            case 5:
                h ^= ui64(data2[4]) << 32;
                [[fallthrough]];

            case 4:
                h ^= ui64(data2[3]) << 24;
                [[fallthrough]];

            case 3:
                h ^= ui64(data2[2]) << 16;
                [[fallthrough]];

            case 2:
                h ^= ui64(data2[1]) << 8;
                [[fallthrough]];

            case 1:
                h ^= ui64(data2[0]);
                h *= m;
                break;
        }

        h ^= h >> r;
        h *= m;
        h ^= h >> r;

        return h;
    }
} // namespace NMurmurPrivate

template size_t MurmurHash<size_t>(const void* buf, size_t len) noexcept;
