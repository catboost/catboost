#pragma once

#include <util/system/defaults.h>
#include <util/system/unaligned_mem.h>

/*
 * https://sites.google.com/site/murmurhash/
 */

namespace NMurmurPrivate {
    template <size_t>
    struct TMurmurHash2ATraits;

    template <>
    struct TMurmurHash2ATraits<32> {
        using TValue = ui32;
        static const TValue Multiplier = 0x5bd1e995;
        enum { R1 = 24,
               R2 = 13,
               R3 = 15 };
    };

    template <>
    struct TMurmurHash2ATraits<64> {
        using TValue = ui64;
        static const TValue Multiplier = ULL(0xc6a4a7935bd1e995);
        enum { R1 = 47,
               R2 = 47,
               R3 = 47 };
    };
}

template <class T>
class TMurmurHash2A {
private:
    using TTraits = typename NMurmurPrivate::TMurmurHash2ATraits<8 * sizeof(T)>;
    using TValue = typename TTraits::TValue;

public:
    inline TMurmurHash2A(TValue seed = 0)
        : Hash(seed)
    {
    }

    inline TMurmurHash2A& Update(const void* buf, size_t len) noexcept {
        Size += len;

        MixTail(buf, len);

        while (len >= sizeof(TValue)) {
            Hash = Mix(Hash, ReadUnaligned<TValue>(buf));
            buf = static_cast<const char*>(buf) + sizeof(TValue);
            len -= sizeof(TValue);
        }

        MixTail(buf, len);

        return *this;
    }

    inline TValue Value() const noexcept {
        TValue hash = Mix(Mix(Hash, Tail), (TValue)Size);

        hash ^= hash >> TTraits::R2;
        hash *= TTraits::Multiplier;
        hash ^= hash >> TTraits::R3;

        return hash;
    }

private:
    static inline TValue Mix(TValue h, TValue k) noexcept {
        k *= TTraits::Multiplier;
        k ^= k >> TTraits::R1;
        k *= TTraits::Multiplier;
        h *= TTraits::Multiplier;
        h ^= k;
        return h;
    }

    inline void MixTail(const void*& buf, size_t& len) noexcept {
        while (len && (len < sizeof(TValue) || Count)) {
            Tail |= (TValue) * ((const unsigned char*&)buf)++ << (Count++ * 8);

            --len;

            if (Count == sizeof(TValue)) {
                Hash = Mix(Hash, Tail);
                Tail = 0;
                Count = 0;
            }
        }
    }

private:
    TValue Hash = 0;
    TValue Tail = 0;
    size_t Count = 0;
    size_t Size = 0;
};
