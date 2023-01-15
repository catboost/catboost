#include <library/cpp/l2_distance/l2_distance.h>
#include <library/cpp/testing/benchmark/bench.h>

#include <contrib/libs/eigen/Eigen/Core>

#include <util/generic/singleton.h>
#include <util/generic/vector.h>
#include <util/random/fast.h>

namespace {
    inline void RandomNumber(TReallyFastRng32& rng, i8& value) {
        value = rng.Uniform(~((ui8)0));
    };

    inline void RandomNumber(TReallyFastRng32& rng, ui8& value) {
        value = rng.Uniform(~((ui8)0));
    };

    inline void RandomNumber(TReallyFastRng32& rng, i32& value) {
        value = rng.Uniform(~((ui32)0));
    };

    inline void RandomNumber(TReallyFastRng32& rng, ui32& value) {
        value = rng.Uniform(~((ui32)0));
    };

    inline void RandomNumber(TReallyFastRng32& rng, float& value) {
        value = rng.GenRandReal1();
    };

    inline void RandomNumber(TReallyFastRng32& rng, double& value) {
        value = rng.GenRandReal1();
    };

    template <class Result, class Number>
    class TResultType {
    public:
        using TNumber = Number;
        using TResult = Result;
        using TIResult = Result;
    };

    template <>
    class TResultType<ui32, i8> {
    public:
        using TNumber = i8;
        using TResult = ui32;
        using TIResult = i32;
    };

    template <>
    class TResultType<ui32, ui8> {
    public:
        using TNumber = i8;
        using TResult = ui32;
        using TIResult = i32;
    };

    template <>
    class TResultType<ui64, i32> {
    public:
        using TNumber = i32;
        using TResult = ui64;
        using TIResult = i64;
    };

    template <>
    class TResultType<ui64, ui32> {
    public:
        using TNumber = i32;
        using TResult = ui64;
        using TIResult = i64;
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

    template <typename Result, typename Number>
    Result SimpleL2Distance(const Number* lhs, const Number* rhs, int length) {
        Result sum = 0;
        for (int i = 0; i < length; ++i) {
            Result diff = lhs[i] < rhs[i] ? rhs[i] - lhs[i] : lhs[i] - rhs[i];
            sum += diff * diff;
        }
        return sum;
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
                    Y_DO_NOT_OPTIMIZE_AWAY(op(lhs + start, rhs + start, length - start));
                }
            }
        }

    private:
        TData1 Data1_;
        TData2 Data2_;
    };

    TBenchmark<ui32, i8, 1000, 1, 179> Bench1000_i8;
    TBenchmark<ui32, ui8, 1000, 413, 19> Bench1000_ui8;
    TBenchmark<ui64, i32, 1000, 19, 117> Bench1000_i32;
    TBenchmark<ui64, ui32, 1000, 149, 17> Bench1000_ui32;
    TBenchmark<float, float, 1000, 19, 117> Bench1000_float;
    TBenchmark<double, double, 1000, 19, 117> Bench1000_double;

    TBenchmark<ui32, i8, 30000, 1, 179> Bench30000_i8;
    TBenchmark<ui32, ui8, 30000, 1, 179> Bench30000_ui8;
    TBenchmark<ui64, i32, 30000, 19, 117> Bench30000_i32;
    TBenchmark<ui64, ui32, 30000, 19, 117> Bench30000_ui32;
    TBenchmark<float, float, 30000, 19, 117> Bench30000_float;
    TBenchmark<double, double, 30000, 19, 117> Bench30000_double;

    /* 8-bit: */
    Y_CPU_BENCHMARK(Naive1000_i8, iface) {
        Bench1000_i8.Do(SimpleL2Distance, iface);
    }

    Y_CPU_BENCHMARK(Slow1000_i8, iface) {
        Bench1000_i8.Do(L2DistanceSlow, iface);
    }

    Y_CPU_BENCHMARK(Fast1000_i8, iface) {
        Bench1000_i8.Do(L2Distance, iface);
    }

    Y_CPU_BENCHMARK(Naive1000_ui8, iface) {
        Bench1000_ui8.Do(SimpleL2Distance, iface);
    }

    Y_CPU_BENCHMARK(Slow1000_ui8, iface) {
        Bench1000_ui8.Do(L2DistanceSlow, iface);
    }

    Y_CPU_BENCHMARK(Fast1000_ui8, iface) {
        Bench1000_ui8.Do(L2Distance, iface);
    }

    /* 32-bit: */
    Y_CPU_BENCHMARK(Naive1000_i32, iface) {
        Bench1000_i32.Do(SimpleL2Distance, iface);
    }

    Y_CPU_BENCHMARK(Slow1000_i32, iface) {
        Bench1000_i32.Do(L2DistanceSlow, iface);
    }

    Y_CPU_BENCHMARK(Fast1000_i32, iface) {
        Bench1000_i32.Do(L2Distance, iface);
    }

    Y_CPU_BENCHMARK(Naive1000_ui32, iface) {
        Bench1000_ui32.Do(SimpleL2Distance, iface);
    }

    Y_CPU_BENCHMARK(Slow1000_ui32, iface) {
        Bench1000_ui32.Do(L2DistanceSlow, iface);
    }

    Y_CPU_BENCHMARK(Fast1000_ui32, iface) {
        Bench1000_ui32.Do(L2Distance, iface);
    }

    /* float: */
    Y_CPU_BENCHMARK(Naive1000_float, iface) {
        Bench1000_float.Do(SimpleL2Distance, iface);
    }

    Y_CPU_BENCHMARK(Slow1000_float, iface) {
        Bench1000_float.Do(L2DistanceSlow, iface);
    }

    Y_CPU_BENCHMARK(Fast1000_float, iface) {
        Bench1000_float.Do(L2Distance, iface);
    }

    /* double: */
    Y_CPU_BENCHMARK(Naive1000_double, iface) {
        Bench1000_double.Do(SimpleL2Distance, iface);
    }

    Y_CPU_BENCHMARK(Slow1000_double, iface) {
        Bench1000_double.Do(L2DistanceSlow, iface);
    }

    Y_CPU_BENCHMARK(Fast1000_double, iface) {
        Bench1000_double.Do(L2Distance, iface);
    }

    /************* 30000 tests ****************/

    Y_CPU_BENCHMARK(Naive30000_i8, iface) {
        Bench30000_i8.Do(SimpleL2Distance, iface);
    }

    Y_CPU_BENCHMARK(Slow30000_i8, iface) {
        Bench30000_i8.Do(L2SqrDistanceSlow, iface);
    }

    Y_CPU_BENCHMARK(Fast30000_i8, iface) {
        Bench30000_i8.Do(L2SqrDistance, iface);
    }

    Y_CPU_BENCHMARK(Naive30000_ui8, iface) {
        Bench30000_ui8.Do(SimpleL2Distance, iface);
    }

    Y_CPU_BENCHMARK(Slow30000_ui8, iface) {
        Bench30000_ui8.Do(L2SqrDistanceSlow, iface);
    }

    Y_CPU_BENCHMARK(Fast30000_ui8, iface) {
        Bench30000_ui8.Do(L2SqrDistance, iface);
    }

    Y_CPU_BENCHMARK(Naive30000_i32, iface) {
        Bench30000_i32.Do(SimpleL2Distance, iface);
    }

    Y_CPU_BENCHMARK(Slow30000_i32, iface) {
        Bench30000_i32.Do(L2SqrDistanceSlow, iface);
    }

    Y_CPU_BENCHMARK(Fast30000_i32, iface) {
        Bench30000_i32.Do(L2SqrDistance, iface);
    }

    Y_CPU_BENCHMARK(Naive30000_ui32, iface) {
        Bench30000_ui32.Do(SimpleL2Distance, iface);
    }

    Y_CPU_BENCHMARK(Slow30000_ui32, iface) {
        Bench30000_ui32.Do(L2SqrDistanceSlow, iface);
    }

    Y_CPU_BENCHMARK(Fast30000_ui32, iface) {
        Bench30000_ui32.Do(L2SqrDistance, iface);
    }

    Y_CPU_BENCHMARK(Naive30000_float, iface) {
        Bench30000_float.Do(SimpleL2Distance, iface);
    }

    Y_CPU_BENCHMARK(Slow30000_float, iface) {
        Bench30000_float.Do(L2SqrDistanceSlow, iface);
    }

    Y_CPU_BENCHMARK(Fast30000_float, iface) {
        Bench30000_float.Do(L2SqrDistance, iface);
    }

    Y_CPU_BENCHMARK(Naive30000_double, iface) {
        Bench30000_double.Do(SimpleL2Distance, iface);
    }

    Y_CPU_BENCHMARK(Slow30000_double, iface) {
        Bench30000_double.Do(L2SqrDistanceSlow, iface);
    }

    Y_CPU_BENCHMARK(Fast30000_double, iface) {
        Bench30000_double.Do(L2SqrDistance, iface);
    }

}
