#include "normal.h"
#include "common_ops.h"
#include "random.h"

namespace {
    template <class T>
    struct TSysRNG: public TCommonRNG<T, TSysRNG<T>> {
        inline T GenRand() noexcept {
            return RandomNumber<T>();
        }
    };
}

template <>
float StdNormalRandom<float>() noexcept {
    return StdNormalDistribution<float>(TSysRNG<ui64>());
}

template <>
double StdNormalRandom<double>() noexcept {
    return StdNormalDistribution<double>(TSysRNG<ui64>());
}

template <>
long double StdNormalRandom<long double>() noexcept {
    return StdNormalDistribution<long double>(TSysRNG<ui64>());
}
