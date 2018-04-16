#include <library/dot_product/dot_product.h>

#include <library/testing/benchmark/bench.h>

#include <util/generic/singleton.h>
#include <util/generic/vector.h>
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
    public:
        TRandomDataHolder() {
            TReallyFastRng32 rng(seed);
            for (size_t i = 0; i < length; ++i) {
                RandomNumber(rng, Data_[i]);
            }
        }

        const Number* Data() const {
            return Data_;
        }

        int Length() const {
            return length;
        }

    private:
        alignas(16) Number Data_[length];
    };

    using RDH_1000_1 = TRandomDataHolder<i8, 1000, 1>;
    using RDH_1000_2 = TRandomDataHolder<i8, 1000, 179>;
    using RDH_30000_1 = TRandomDataHolder<i8, 30000, 1>;
    using RDH_30000_2 = TRandomDataHolder<i8, 30000, 179>;

    using RDH32_1000_1 = TRandomDataHolder<i32, 1000, 1>;
    using RDH32_1000_2 = TRandomDataHolder<i32, 1000, 217>;
    using RDH32_30000_1 = TRandomDataHolder<i32, 30000, 1>;
    using RDH32_30000_2 = TRandomDataHolder<i32, 30000, 217>;

    using RDHF_1000_1 = TRandomDataHolder<float, 1000, 1>;
    using RDHF_1000_2 = TRandomDataHolder<float, 1000, 117>;
    using RDHF_30000_1 = TRandomDataHolder<float, 30000, 1>;
    using RDHF_30000_2 = TRandomDataHolder<float, 30000, 117>;

    using RDHD_1000_1 = TRandomDataHolder<double, 1000, 13>;
    using RDHD_1000_2 = TRandomDataHolder<double, 1000, 65537>;
    using RDHD_30000_1 = TRandomDataHolder<double, 30000, 13>;
    using RDHD_30000_2 = TRandomDataHolder<double, 30000, 65537>;

    template <typename Res, typename Num>
    Res SimpleDotProduct(const Num* lhs, const Num* rhs, int length) {
        Res sum = 0;

        while (length) {
            sum += static_cast<Res>(*lhs++) * static_cast<Res>(*rhs++);
            --length;
        }

        return sum;
    }

    template <typename Res, typename Num>
    Res EigenDotProduct(const Num* lhs, const Num* rhs, int length) {
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

        void Do(Res (*op)(const Number*, const Number*, int), const NBench::NCpu::TParams& iface) {
            int length = Data1_.Length();
            const Number* lhs = Data1_.Data();
            const Number* rhs = Data2_.Data();

            for (size_t i = 0; i < iface.Iterations(); ++i) {
                Y_UNUSED(i);
                for (int start = 0; start + 100 <= length; start += 16) {
                    Y_DO_NOT_OPTIMIZE_AWAY(op(lhs + start, rhs + start, length));
                }
            }
        }

    private:
        TData1 Data1_;
        TData2 Data2_;
    };

    static TBenchmark<i32, i8, 1000, 1, 179> Bench1000_i8;
    static TBenchmark<i64, i32, 1000, 19, 117> Bench1000_i32;
    static TBenchmark<float, float, 1000, 19, 117> Bench1000_float;
    static TBenchmark<double, double, 1000, 19, 117> Bench1000_double;

    static TBenchmark<i32, i8, 30000, 1, 179> Bench30000_i8;
    static TBenchmark<i64, i32, 30000, 19, 117> Bench30000_i32;
    static TBenchmark<float, float, 30000, 19, 117> Bench30000_float;
    static TBenchmark<double, double, 30000, 19, 117> Bench30000_double;

    /* 8-bit: */
    Y_CPU_BENCHMARK(Slow1000_8, iface) {
        Bench1000_i8.Do(SimpleDotProduct, iface);
    }

    Y_CPU_BENCHMARK(Eigen1000_8, iface) {
        Bench1000_i8.Do(EigenDotProduct, iface);
    }

    Y_CPU_BENCHMARK(Vector1000_8, iface) {
        Bench1000_i8.Do(DotProductSlow, iface);
    }

    Y_CPU_BENCHMARK(Fast1000_8, iface) {
        Bench1000_i8.Do(DotProduct, iface);
    }

    /* 32-bit: */
    Y_CPU_BENCHMARK(Slow1000_32, iface) {
        Bench1000_i32.Do(SimpleDotProduct, iface);
    }

    Y_CPU_BENCHMARK(Eigen1000_32, iface) {
        Bench1000_i32.Do(EigenDotProduct, iface);
    }

    Y_CPU_BENCHMARK(Vector1000_32, iface) {
        Bench1000_i32.Do(DotProductSlow, iface);
    }

    Y_CPU_BENCHMARK(Fast1000_32, iface) {
        Bench1000_i32.Do(DotProduct, iface);
    }

    /* float: */
    Y_CPU_BENCHMARK(Slow1000_float, iface) {
        Bench1000_float.Do(SimpleDotProduct, iface);
    }

    Y_CPU_BENCHMARK(Eigen1000_float, iface) {
        Bench1000_float.Do(EigenDotProduct, iface);
    }

    Y_CPU_BENCHMARK(Vector1000_float, iface) {
        Bench1000_float.Do(DotProductSlow, iface);
    }

    Y_CPU_BENCHMARK(Fast1000_float, iface) {
        Bench1000_float.Do(DotProduct, iface);
    }

    /* double: */
    Y_CPU_BENCHMARK(Slow1000_double, iface) {
        Bench1000_double.Do(SimpleDotProduct, iface);
    }

    Y_CPU_BENCHMARK(Eigen1000_double, iface) {
        Bench1000_double.Do(EigenDotProduct, iface);
    }

    Y_CPU_BENCHMARK(Vector1000_double, iface) {
        Bench1000_double.Do(DotProductSlow, iface);
    }

    Y_CPU_BENCHMARK(Fast1000_double, iface) {
        Bench1000_double.Do(DotProduct, iface);
    }

    Y_CPU_BENCHMARK(Slow30000_8, iface) {
        Bench30000_i8.Do(SimpleDotProduct, iface);
    }

    Y_CPU_BENCHMARK(Eigen30000_8, iface) {
        Bench30000_i8.Do(EigenDotProduct, iface);
    }

    Y_CPU_BENCHMARK(Vector30000_8, iface) {
        Bench30000_i8.Do(DotProductSlow, iface);
    }

    Y_CPU_BENCHMARK(Fast30000_8, iface) {
        Bench30000_i8.Do(DotProduct, iface);
    }

    Y_CPU_BENCHMARK(Slow30000_32, iface) {
        Bench30000_i32.Do(SimpleDotProduct, iface);
    }

    Y_CPU_BENCHMARK(Eigen30000_32, iface) {
        Bench30000_i32.Do(EigenDotProduct, iface);
    }

    Y_CPU_BENCHMARK(Vector30000_32, iface) {
        Bench30000_i32.Do(DotProductSlow, iface);
    }

    Y_CPU_BENCHMARK(Fast30000_32, iface) {
        Bench30000_i32.Do(DotProduct, iface);
    }

    Y_CPU_BENCHMARK(Slow30000_float, iface) {
        Bench30000_float.Do(SimpleDotProduct, iface);
    }

    Y_CPU_BENCHMARK(Eigen30000_float, iface) {
        Bench30000_float.Do(EigenDotProduct, iface);
    }

    Y_CPU_BENCHMARK(Vector30000_float, iface) {
        Bench30000_float.Do(DotProductSlow, iface);
    }

    Y_CPU_BENCHMARK(Fast30000_float, iface) {
        Bench30000_float.Do(DotProduct, iface);
    }

    Y_CPU_BENCHMARK(Slow30000_double, iface) {
        Bench30000_double.Do(SimpleDotProduct, iface);
    }

    Y_CPU_BENCHMARK(Eigen30000_double, iface) {
        Bench30000_double.Do(EigenDotProduct, iface);
    }

    Y_CPU_BENCHMARK(Vector30000_double, iface) {
        Bench30000_double.Do(DotProductSlow, iface);
    }

    Y_CPU_BENCHMARK(Fast30000_double, iface) {
        Bench30000_double.Do(DotProduct, iface);
    }

}
