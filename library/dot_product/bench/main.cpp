#include <library/dot_product/dot_product.h>

#include <library/testing/benchmark/bench.h>

#include <util/generic/singleton.h>
#include <util/generic/vector.h>
#include <util/generic/xrange.h>
#include <contrib/libs/eigen/Eigen/Core>

#include <util/random/fast.h>

namespace {
    inline void RandomNumber(TReallyFastRng32& rng, i8& value) {
        value = rng.Uniform(~((ui8)0));
    };

    inline void RandomNumber(TReallyFastRng32& rng, i32& value) {
        value = rng.Uniform(~((ui32)0));
    };

    inline void RandomNumber(TReallyFastRng32& rng, float& value) {
        value = rng.GenRandReal1();
    };

    inline void RandomNumber(TReallyFastRng32& rng, double& value) {
        value = rng.GenRandReal1();
    };

    template <class Number, size_t length, size_t seed>
    class TRandomDataHolder {
        static constexpr const size_t SequenceLength = Max(length * 2, (size_t)4096);

    public:
        TRandomDataHolder() {
            TReallyFastRng32 rng(seed);
            rng.Advance(length);
            for (Number& n : Data_) {
                RandomNumber(rng, n);
            }
        }

        /// Возвращает набор смещений, каждое из которых можно использовать как начало окна с длиной `length`
        static auto SlidingWindows() {
            static_assert(SequenceLength >= length);
            return xrange((size_t)0, SequenceLength - length, 16);
        }

        const Number* Data() const {
            return Data_;
        }

        ui32 Length() const {
            return length;
        }

    private:
        char Padding0[64];                        // memory sanitizer guard range
        alignas(16) Number Data_[SequenceLength]; // в тесте используется скользящее окно с длиной length, которое имеет смещение до length, поэтому адресуются length*2 значений
        char Padding1[64];                        // memory sanitizer guard range
    };

    template <typename Res, typename Num>
    Res SimpleDotProduct(const Num* lhs, const Num* rhs, ui32 length) {
        Res sum = 0;

        while (length) {
            sum += static_cast<Res>(*lhs++) * static_cast<Res>(*rhs++);
            --length;
        }

        return sum;
    }

    template <typename Res, typename Num>
    Res EigenDotProduct(const Num* lhs, const Num* rhs, ui32 length) {
        Eigen::Map<const Eigen::Matrix<Num, 1, Eigen::Dynamic>> el(lhs, length);
        Eigen::Map<const Eigen::Matrix<Num, Eigen::Dynamic, 1>> er(rhs, length);
        Res res[1];
        Eigen::Map<Eigen::Matrix<Res, 1, 1>> r(res);

        r = el.template cast<Res>() * er.template cast<Res>();

        return res[0];
    }

    template <typename Res, typename Number, size_t count, size_t seed1, size_t seed2>
    class TBenchmark {
    public:
        using TData1 = TRandomDataHolder<Number, count, seed1>;
        using TData2 = TRandomDataHolder<Number, count, seed2>;

        TBenchmark() {
        }

        void Do(Res (*op)(const Number*, const Number*, ui32), const NBench::NCpu::TParams& iface) {
            ui32 length = Data1_.Length();
            const Number* lhs = Data1_.Data();
            const Number* rhs = Data2_.Data();

            for (size_t i = 0; i < iface.Iterations(); ++i) {
                for (ui32 start : TData1::SlidingWindows()) {
                    Y_DO_NOT_OPTIMIZE_AWAY(op(lhs + start, rhs + start, length));
                }
            }
        }

    private:
        TData1 Data1_;
        TData2 Data2_;
    };

    template <typename TSourceType>
    struct TResultType {
        using TType = TSourceType;
    };

    template <>
    struct TResultType<i8> {
        using TType = i32;
    };

    template <>
    struct TResultType<i32> {
        using TType = i64;
    };

#define DefineBenchmarkAlgos(length, TSourceType)                                                                   \
    static TBenchmark<TResultType<TSourceType>::TType, TSourceType, length, 19, 179> Bench##length##_##TSourceType; \
                                                                                                                    \
    Y_CPU_BENCHMARK(Slow##length##_##TSourceType, iface) {                                                          \
        Bench##length##_##TSourceType.Do(SimpleDotProduct, iface);                                                  \
    }                                                                                                               \
    Y_CPU_BENCHMARK(Eigen##length##_##TSourceType, iface) {                                                         \
        Bench##length##_##TSourceType.Do(EigenDotProduct, iface);                                                   \
    }                                                                                                               \
    Y_CPU_BENCHMARK(Vector##length##_##TSourceType, iface) {                                                        \
        Bench##length##_##TSourceType.Do(DotProductSlow, iface);                                                    \
    }                                                                                                               \
    Y_CPU_BENCHMARK(Fast##length##_##TSourceType, iface) {                                                          \
        Bench##length##_##TSourceType.Do(DotProduct, iface);                                                        \
    }

#define DefineBenchmarkLengths(TSourceType)  \
    DefineBenchmarkAlgos(32, TSourceType);   \
    DefineBenchmarkAlgos(96, TSourceType);   \
    DefineBenchmarkAlgos(100, TSourceType);  \
    DefineBenchmarkAlgos(150, TSourceType);  \
    DefineBenchmarkAlgos(200, TSourceType);  \
    DefineBenchmarkAlgos(350, TSourceType);  \
    DefineBenchmarkAlgos(700, TSourceType);  \
    DefineBenchmarkAlgos(1000, TSourceType); \
    DefineBenchmarkAlgos(30000, TSourceType);

    DefineBenchmarkLengths(i8);
    DefineBenchmarkLengths(i32);
    DefineBenchmarkLengths(float);
    DefineBenchmarkLengths(double);
}
