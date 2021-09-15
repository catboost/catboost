#include <library/cpp/dot_product/dot_product.h>
#include <library/cpp/dot_product/dot_product_sse.h>
#include <library/cpp/dot_product/dot_product_avx2.h>
#include <library/cpp/dot_product/dot_product_simple.h>

#include <benchmark/benchmark.h>

#include <contrib/libs/eigen/Eigen/Core>

#include <util/generic/singleton.h>
#include <util/generic/vector.h>
#include <util/generic/xrange.h>
#include <util/generic/ymath.h>
#include <util/random/fast.h>

constexpr size_t SlidingWindowStride = 16;

template <class T>
TVector<T> MakeRandomData(size_t size, ui32 seed) {
    const size_t length = CeilDiv(Max<size_t>(size * 2, 4096), SlidingWindowStride) * SlidingWindowStride;
    TReallyFastRng32 rng(seed);
    rng.Advance(length);

    TVector<T> result(length);
    for (auto& n : result) {
        if constexpr (std::is_integral_v<T>) {
            const auto x = rng.GenRand();
            static_assert(sizeof(T) <= sizeof(x));
            memcpy(&n, &x, sizeof(T));
        } else {
            n = rng.GenRandReal1();
        }
    }

    return result;
}

struct TSimpleDotProduct {
    template <class T>
    auto operator()(const T* lhs, const T* rhs, size_t length) {
        return DotProductSimple(lhs, rhs, length);
    }
};

struct TEigenDotProduct {
    template <class T>
    auto operator()(const T* lhs, const T* rhs, size_t length) {
        Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, 1>> el(lhs, length);
        Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, 1>> er(rhs, length);

        return (el * er).sum();
    }
};

struct TSseDotProduct {
    template <class T>
    auto operator()(const T* lhs, const T* rhs, size_t length) {
        return DotProductSse(lhs, rhs, length);
    }
};

struct TAvx2DotProduct {
    template <class T>
    auto operator()(const T* lhs, const T* rhs, size_t length) {
        return DotProductAvx2(lhs, rhs, length);
    }
};

struct TDotProduct {
    template <class T>
    auto operator()(const T* lhs, const T* rhs, size_t length) {
        return DotProduct(lhs, rhs, length);
    }
};

struct TSequentialCosine {
    template <class T>
    auto operator()(const T* lhs, const T* rhs, size_t length) {
        const auto ll = L2NormSquared(lhs, length);
        const auto lr = DotProduct(lhs, rhs, length);
        const auto rr = L2NormSquared(rhs, length);
        return lr / sqrt(ll * rr);
    }
};

struct TCombinedCosine {
    template <class T>
    auto operator()(const T* lhs, const T* rhs, size_t length) {
        const auto p = TriWayDotProduct(lhs, rhs, length, ETriWayDotProductComputeMask::All);
        return p.LR / sqrt(p.LL * p.RR);
    }
};

struct TSequentialCosinePrenorm {
    template <class T>
    auto operator()(const T* lhs, const T* rhs, size_t length) {
        const auto ll = L2NormSquared(lhs, length);
        const auto lr = DotProduct(lhs, rhs, length);
        return lr / sqrt(ll);
    }
};

struct TCombinedCosinePrenorm {
    template <class T>
    auto operator()(const T* lhs, const T* rhs, size_t length) {
        const auto p = TriWayDotProduct(lhs, rhs, length, ETriWayDotProductComputeMask::Left);
        return p.LR / sqrt(p.LL);
    }
};

constexpr size_t Seed1 = 19;
constexpr size_t Seed2 = 179;

template <class T, class TDotProduct>
void BM_DotProduct(benchmark::State& state) {
    const size_t size = state.range(0);
    const auto data1 = MakeRandomData<T>(size, Seed1);
    const auto data2 = MakeRandomData<T>(size, Seed2);
    const auto dataSize = data1.size();
    const auto* lhs = data1.data();
    const auto* rhs = data2.data();

    size_t offset = 0;
    for (const auto _: state) {
        benchmark::DoNotOptimize(TDotProduct{}(lhs + offset, rhs + offset, size));
        offset += SlidingWindowStride;
        if (offset + size > dataSize) {
            offset = 0;
        }
    }
}

void WithSizes(benchmark::internal::Benchmark* b) {
    for (const auto i: {32, 48, 50, 64, 96, 100, 150, 200, 350, 700, 1024, 30000}) {
        b->Arg(i);
    }
}

BENCHMARK_TEMPLATE(BM_DotProduct, i8, TSseDotProduct)->Apply(WithSizes);
BENCHMARK_TEMPLATE(BM_DotProduct, ui8, TSseDotProduct)->Apply(WithSizes);
BENCHMARK_TEMPLATE(BM_DotProduct, i32, TSseDotProduct)->Apply(WithSizes);
BENCHMARK_TEMPLATE(BM_DotProduct, float, TSseDotProduct)->Apply(WithSizes);
BENCHMARK_TEMPLATE(BM_DotProduct, double, TSseDotProduct)->Apply(WithSizes);

BENCHMARK_TEMPLATE(BM_DotProduct, i8, TAvx2DotProduct)->Apply(WithSizes);
BENCHMARK_TEMPLATE(BM_DotProduct, ui8, TAvx2DotProduct)->Apply(WithSizes);
BENCHMARK_TEMPLATE(BM_DotProduct, i32, TAvx2DotProduct)->Apply(WithSizes);
BENCHMARK_TEMPLATE(BM_DotProduct, float, TAvx2DotProduct)->Apply(WithSizes);
BENCHMARK_TEMPLATE(BM_DotProduct, double, TAvx2DotProduct)->Apply(WithSizes);

BENCHMARK_TEMPLATE(BM_DotProduct, i8, TDotProduct)->Apply(WithSizes);
BENCHMARK_TEMPLATE(BM_DotProduct, ui8, TDotProduct)->Apply(WithSizes);
BENCHMARK_TEMPLATE(BM_DotProduct, i32, TDotProduct)->Apply(WithSizes);
BENCHMARK_TEMPLATE(BM_DotProduct, float, TDotProduct)->Apply(WithSizes);
BENCHMARK_TEMPLATE(BM_DotProduct, double, TDotProduct)->Apply(WithSizes);

BENCHMARK_TEMPLATE(BM_DotProduct, float, TSequentialCosine)->Apply(WithSizes);
BENCHMARK_TEMPLATE(BM_DotProduct, float, TCombinedCosine)->Apply(WithSizes);
BENCHMARK_TEMPLATE(BM_DotProduct, float, TSequentialCosinePrenorm)->Apply(WithSizes);
BENCHMARK_TEMPLATE(BM_DotProduct, float, TCombinedCosinePrenorm)->Apply(WithSizes);

BENCHMARK_TEMPLATE(BM_DotProduct, i8, TSimpleDotProduct)->Apply(WithSizes);
BENCHMARK_TEMPLATE(BM_DotProduct, ui8, TSimpleDotProduct)->Apply(WithSizes);
BENCHMARK_TEMPLATE(BM_DotProduct, i32, TSimpleDotProduct)->Apply(WithSizes);
BENCHMARK_TEMPLATE(BM_DotProduct, float, TSimpleDotProduct)->Apply(WithSizes);
BENCHMARK_TEMPLATE(BM_DotProduct, double, TSimpleDotProduct)->Apply(WithSizes);

BENCHMARK_TEMPLATE(BM_DotProduct, i8, TEigenDotProduct)->Apply(WithSizes);
BENCHMARK_TEMPLATE(BM_DotProduct, ui8, TEigenDotProduct)->Apply(WithSizes);
BENCHMARK_TEMPLATE(BM_DotProduct, i32, TEigenDotProduct)->Apply(WithSizes);
BENCHMARK_TEMPLATE(BM_DotProduct, float, TEigenDotProduct)->Apply(WithSizes);
BENCHMARK_TEMPLATE(BM_DotProduct, double, TEigenDotProduct)->Apply(WithSizes);

