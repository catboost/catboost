#include "restorable_rng.h"

TVector<ui64> GenRandUI64Vector(int size, ui64 randomSeed) {
    TFastRng64 rand(randomSeed);
    TVector<ui64> result(size);
    for (auto& value : result) {
        value = rand.GenRand();
    }
    return result;
}
