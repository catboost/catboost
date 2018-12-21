#pragma once

#include <util/random/fast.h>
#include <util/ysaveload.h>
#include <util/generic/vector.h>

struct TRestorableFastRng64 : public TCommonRNG<ui64, TRestorableFastRng64> {
    template <typename T>
    TRestorableFastRng64(T&& seedSource)
        : SeedArgs(std::forward<T>(seedSource))
        , Rng(SeedArgs)
    {
    }

    inline void Save(IOutputStream* s) const {
        ::SaveMany(
            s,
            SeedArgs.Seed1,
            SeedArgs.Seed2,
            SeedArgs.Seq1,
            SeedArgs.Seq2,
            CallCount);
    }
    inline void Load(IInputStream* s) {
        ::LoadMany(
            s,
            SeedArgs.Seed1,
            SeedArgs.Seed2,
            SeedArgs.Seq1,
            SeedArgs.Seq2,
            CallCount);
        new (&Rng) TFastRng64(SeedArgs);
        if (CallCount > 0) {
            Rng.Advance(CallCount);
        }
    }
    inline ui64 GenRand() noexcept {
        ++CallCount;
        return Rng.GenRand();
    }

    inline void Advance(ui64 delta) noexcept {
        CallCount += delta;
        Rng.Advance(delta);
    }

    ui64 GetCallCount() const {
        return CallCount;
    }
private:
    TFastRng64::TArgs SeedArgs;
    TFastRng64 Rng;
    ui64 CallCount = 0;
};

TVector<ui64> GenRandUI64Vector(int size, ui64 randomSeed);
