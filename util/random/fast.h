#pragma once

#include "lcg_engine.h"
#include "common_ops.h"

#include <util/generic/bitops.h>
#include <util/system/platform.h>

// based on http://www.pcg-random.org/. See T*FastRng* family below.

struct TPCGMixer {
    static constexpr ui32 Mix(ui64 x) noexcept {
        const ui32 xorshifted = ((x >> 18u) ^ x) >> 27u;
        const ui32 rot = x >> 59u;

        return RotateBitsRight(xorshifted, rot);
    }
};

using TFastRng32Base = TLcgRngBase<TLcgIterator<ui64, ULL(6364136223846793005)>, TPCGMixer>;
using TReallyFastRng32Base = TLcgRngBase<TFastLcgIterator<ui64, ULL(6364136223846793005), ULL(1)>, TPCGMixer>;

class IInputStream;

struct TFastRng32: public TCommonRNG<ui32, TFastRng32>, public TFastRng32Base {
    inline TFastRng32(ui64 seed, ui32 seq)
        : TFastRng32Base(seed, seq)
    {
    }

    TFastRng32(IInputStream& entropy);
};

// faster than TFastRng32, but have only one possible stream sequence
struct TReallyFastRng32: public TCommonRNG<ui32, TReallyFastRng32>, public TReallyFastRng32Base {
    constexpr TReallyFastRng32(ui64 seed)
        : TReallyFastRng32Base(seed)
    {
    }

    TReallyFastRng32(IInputStream& entropy);
};

class TFastRng64: public TCommonRNG<ui64, TFastRng64> {
public:
    struct TArgs {
        constexpr TArgs(ui64 seed) noexcept {
            TReallyFastRng32 rng(seed);

            Seed1 = rng.GenRand64();
            Seq1 = rng.GenRand();
            Seed2 = rng.GenRand64();
            Seq2 = rng.GenRand();
        }

        TArgs(IInputStream& entropy);

        ui64 Seed1;
        ui64 Seed2;
        ui32 Seq1;
        ui32 Seq2;
    };

    constexpr TFastRng64(ui64 seed1, ui32 seq1, ui64 seed2, ui32 seq2) noexcept
        : R1_(seed1, seq1)
        , R2_(seed2, FixSeq(seq1, seq2))
    {
    }

    /*
     * simplify constructions like
     *     TFastRng64 rng(17);
     *     TFastRng64 rng(Seek()); //from any IInputStream
     */
    constexpr TFastRng64(const TArgs& args) noexcept
        : TFastRng64(args.Seed1, args.Seq1, args.Seed2, args.Seq2)
    {
    }

    constexpr ui64 GenRand() noexcept {
        const ui64 x = R1_.GenRand();
        const ui64 y = R2_.GenRand();

        return (x << 32) | y;
    }

    inline void Advance(ui64 delta) noexcept {
        R1_.Advance(delta);
        R2_.Advance(delta);
    }

private:
    static constexpr ui32 FixSeq(ui32 seq1, ui32 seq2) noexcept {
        constexpr ui32 mask = ~ui32{0} >> 1;
        return (seq1 & mask) == (seq2 & mask) ? ~seq2 : seq2;
    }

    TFastRng32Base R1_;
    TFastRng32Base R2_;
};

namespace NPrivate {
    template <typename T>
    struct TFastRngTraits;

    template <>
    struct TFastRngTraits<ui32> {
        using TResult = TReallyFastRng32;
    };

    template <>
    struct TFastRngTraits<ui64> {
        using TResult = TFastRng64;
    };
} // namespace NPrivate

template <typename T>
using TFastRng = typename ::NPrivate::TFastRngTraits<T>::TResult;
