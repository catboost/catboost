#include <library/cpp/l1_distance/l1_distance.h>
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

    template <typename T>
    T LDelta(T a, T b) {
        if (a < b)
            return b - a;
        return a - b;
    }

    template <typename Number, typename Result = decltype(L1Distance(static_cast<const Number*>(nullptr), static_cast<const Number*>(nullptr), 0))>
    Result SimpleL1Distance(const Number* lhs, const Number* rhs, int length) {
        Result sum = 0;
        for (int i = 0; i < length; ++i) {
            sum += LDelta(lhs[i], rhs[i]);
        }
        return sum;
    }

    template <typename Number, typename Result = decltype(L1Distance(static_cast<const Number*>(nullptr), static_cast<const Number*>(nullptr), 0))>
    Result VectorL1Distance(const Number* lhs, const Number* rhs, int length) {
        Result s0 = 0;
        Result s1 = 0;
        Result s2 = 0;
        Result s3 = 0;

        while (length >= 4) {
            s0 += LDelta(lhs[0], rhs[0]);
            s1 += LDelta(lhs[1], rhs[1]);
            s2 += LDelta(lhs[2], rhs[2]);
            s3 += LDelta(lhs[3], rhs[3]);
            length -= 4;
            lhs += 4;
            rhs += 4;
        }

        while (length) {
            s0 += LDelta(*lhs++, *rhs++);
            --length;
        }

        return s0 + s1 + s2 + s3;
    }

    // Distance from VisualVocab:
    struct TL1DistanceSSE64ui8_96bytes {
        template <bool MemAligned = false>
        inline ui32 CalcDistance(const ui8* __restrict x, const ui8* __restrict y) {
            Y_ASSERT(!MemAligned || ((x - (const ui8*)nullptr) % 16 == 0 && (y - (const ui8*)nullptr) % 16 == 0));

            __m128i x1 = MemAligned ? _mm_load_si128((__m128i*)&x[0]) : _mm_loadu_si128((__m128i*)&x[0]);
            __m128i y1 = MemAligned ? _mm_load_si128((__m128i*)&y[0]) : _mm_loadu_si128((__m128i*)&y[0]);

            __m128i x2 = MemAligned ? _mm_load_si128((__m128i*)&x[16]) : _mm_loadu_si128((__m128i*)&x[16]);
            __m128i y2 = MemAligned ? _mm_load_si128((__m128i*)&y[16]) : _mm_loadu_si128((__m128i*)&y[16]);

            __m128i sum = _mm_sad_epu8(x1, y1);

            __m128i x3 = MemAligned ? _mm_load_si128((__m128i*)&x[32]) : _mm_loadu_si128((__m128i*)&x[32]);
            __m128i y3 = MemAligned ? _mm_load_si128((__m128i*)&y[32]) : _mm_loadu_si128((__m128i*)&y[32]);

            sum = _mm_add_epi64(sum, _mm_sad_epu8(x2, y2));

            __m128i x4 = MemAligned ? _mm_load_si128((__m128i*)&x[48]) : _mm_loadu_si128((__m128i*)&x[48]);
            __m128i y4 = MemAligned ? _mm_load_si128((__m128i*)&y[48]) : _mm_loadu_si128((__m128i*)&y[48]);

            sum = _mm_add_epi64(sum, _mm_sad_epu8(x3, y3));

            __m128i x5 = MemAligned ? _mm_load_si128((__m128i*)&x[64]) : _mm_loadu_si128((__m128i*)&x[64]);
            __m128i y5 = MemAligned ? _mm_load_si128((__m128i*)&y[64]) : _mm_loadu_si128((__m128i*)&y[64]);

            sum = _mm_add_epi64(sum, _mm_sad_epu8(x4, y4));

            __m128i x6 = MemAligned ? _mm_load_si128((__m128i*)&x[80]) : _mm_loadu_si128((__m128i*)&x[80]);
            __m128i y6 = MemAligned ? _mm_load_si128((__m128i*)&y[80]) : _mm_loadu_si128((__m128i*)&y[80]);

            sum = _mm_add_epi64(sum, _mm_sad_epu8(x5, y5));
            sum = _mm_add_epi64(sum, _mm_sad_epu8(x6, y6));

            return _mm_cvtsi128_si32(sum) + _mm_cvtsi128_si32(_mm_shuffle_epi32(sum, 0xAA));
        }

        inline ui32 operator()(const ui8* __restrict x, const ui8* __restrict y, int) {
            if ((reinterpret_cast<ui64>(x) & 0xF) || (reinterpret_cast<ui64>(y) & 0xF)) {
                return CalcDistance<false>(x, y);
            } else {
                return CalcDistance<true>(x, y);
            }
        }
    };

    template <typename Res, typename Number, size_t count, size_t seed1, size_t seed2>
    class TBenchmark {
    public:
        using TData1 = TRandomDataHolder<Number, count, seed1>;
        using TData2 = TRandomDataHolder<Number, count, seed2>;

        TBenchmark() {
        }

        template <typename F>
        void Do(F&& op, const NBench::NCpu::TParams& iface) {
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

        template <typename F>
        void Do96(F&& op, const NBench::NCpu::TParams& iface) {
            int length = Data1_.Length();
            const Number* lhs = Data1_.Data();
            const Number* rhs = Data2_.Data();

            length -= length % 96 + 96;

            for (size_t i = 0; i < iface.Iterations(); ++i) {
                Y_UNUSED(i);
                for (int start = 0; start < length; start += 16) {
                    Y_DO_NOT_OPTIMIZE_AWAY(op(lhs + start, rhs + start, 96));
                }
            }
        }

    private:
        TData1 Data1_;
        TData2 Data2_;
    };

    TBenchmark<ui32, i8, 1000, 1, 179> Bench1000_i8;
    TBenchmark<ui32, ui8, 1000, 1, 179> Bench1000_ui8;
    TBenchmark<ui64, i32, 1000, 19, 117> Bench1000_i32;
    TBenchmark<ui64, ui32, 1000, 59, 97> Bench1000_ui32;
    TBenchmark<float, float, 1000, 19, 117> Bench1000_float;
    TBenchmark<double, double, 1000, 19, 117> Bench1000_double;

    TBenchmark<ui32, i8, 30000, 1, 179> Bench30000_i8;
    TBenchmark<ui32, ui8, 30000, 1, 179> Bench30000_ui8;
    TBenchmark<ui64, i32, 30000, 19, 117> Bench30000_i32;
    TBenchmark<ui64, ui32, 30000, 37, 131> Bench30000_ui32;
    TBenchmark<float, float, 30000, 19, 117> Bench30000_float;
    TBenchmark<double, double, 30000, 19, 117> Bench30000_double;

#define L_TEST(cnt, type, fcn)                                                                 \
    Y_CPU_BENCHMARK(fcn##_##type##_##cnt, iface) {                                             \
        Bench##cnt##_##type.Do([](auto&& a, auto&& b, int l) { return fcn(a, b, l); }, iface); \
    }

#define L_TEST_96(cnt, type, fcn)                                                                \
    Y_CPU_BENCHMARK(fcn##96##_##type##_##cnt, iface) {                                           \
        Bench##cnt##_##type.Do96([](auto&& a, auto&& b, int l) { return fcn(a, b, l); }, iface); \
    }

#define L_ALL_FUNCS(cnt, type)          \
    L_TEST(cnt, type, SimpleL1Distance) \
    L_TEST(cnt, type, VectorL1Distance) \
    L_TEST(cnt, type, L1DistanceSlow)   \
    L_TEST(cnt, type, L1Distance)

#define L_ALL_TYPES(cnt)    \
    L_ALL_FUNCS(cnt, i8)    \
    L_ALL_FUNCS(cnt, ui8)   \
    L_ALL_FUNCS(cnt, i32)   \
    L_ALL_FUNCS(cnt, ui32)  \
    L_ALL_FUNCS(cnt, float) \
    L_ALL_FUNCS(cnt, double)

    L_TEST_96(30000, ui8, SimpleL1Distance)
    L_TEST_96(30000, ui8, VectorL1Distance)
    L_TEST_96(30000, ui8, L1DistanceSlow)
    L_TEST_96(30000, ui8, L1Distance)
    Y_CPU_BENCHMARK(TL1DistanceSSE64ui8_96bytes_30000, iface) {
        TL1DistanceSSE64ui8_96bytes distance;
        Bench30000_ui8.Do96(distance, iface);
    }
    Y_CPU_BENCHMARK(TL1Distance30000_96_ui8, iface) {
        NL1Distance::TL1Distance<ui8> distance;
        Bench30000_ui8.Do96(distance, iface);
    }

    L_ALL_TYPES(1000)
    L_ALL_TYPES(30000)

}
