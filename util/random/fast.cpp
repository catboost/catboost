#include "fast.h"

#include <util/stream/input.h>

static inline ui32 FixSeq(ui32 seq1, ui32 seq2) noexcept {
    const ui32 mask = (~(ui32)(0)) >> 1;

    if ((seq1 & mask) == (seq2 & mask)) {
        return ~seq2;
    }

    return seq2;
}

TFastRng64::TFastRng64(ui64 seed1, ui32 seq1, ui64 seed2, ui32 seq2) noexcept
    : R1_(seed1, seq1)
    , R2_(seed2, FixSeq(seq1, seq2))
{
}

TFastRng64::TArgs::TArgs(ui64 seed) noexcept {
    TReallyFastRng32 rng(seed);

    Seed1 = rng.GenRand64();
    Seq1 = rng.GenRand();
    Seed2 = rng.GenRand64();
    Seq2 = rng.GenRand();
}

TFastRng64::TArgs::TArgs(IInputStream& entropy) {
    static_assert(sizeof(*this) == 3 * sizeof(ui64), "please, fix me");
    entropy.LoadOrFail(this, sizeof(*this));
}

template <class T>
static inline T Read(IInputStream& in) noexcept {
    T t = T();

    in.LoadOrFail(&t, sizeof(t));

    return t;
}

TFastRng32::TFastRng32(IInputStream& entropy)
    : TFastRng32(Read<ui64>(entropy), Read<ui32>(entropy))
{
}

TReallyFastRng32::TReallyFastRng32(IInputStream& entropy)
    : TReallyFastRng32(Read<ui64>(entropy))
{
}
