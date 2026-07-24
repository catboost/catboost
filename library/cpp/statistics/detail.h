#pragma once

#include <util/generic/typetraits.h>
#include <util/generic/utility.h>
#include <util/generic/yexception.h>
#include <util/generic/ylimits.h>
#include <util/generic/ymath.h>

#include <cmath>
#include <iterator>

namespace NStatistics {
    //! Statistics result information.
    struct TStatTestResult {
        double PValue; /*! pValue test */
        int Sign;      /*! The sign of the deviation of the old sample from the new one */
        TStatTestResult(double pValue = 0, int sign = 0)
            : PValue(pValue)
            , Sign(sign)
        {
        }
    };

    namespace NDetail {
        //! To retrieve underlying type from TKahanAccumulator from library/cpp/accurate_accumulate
        template <bool IsArithmetic, typename T>
        struct TTypeTraits {
            using TResult = typename T::TValueType;
        };

        template <typename TStoreFloatType>
        struct TTypeTraits<true, TStoreFloatType> {
            using TResult = TStoreFloatType;
        };

        //! Relative equal comparator. Returns true if |X - Y| / max{|X|, |Y|} < EPS.
        template <typename ValueType>
        bool RelativeEqual(ValueType x, ValueType y) {
            if (x == 0 && y == 0) {
                return true;
            }
            static const ValueType EPS = 16 * std::numeric_limits<ValueType>::epsilon();

            return fabs(x - y) < EPS * Max(fabs(x), fabs(y));
        }

        template <typename T>
        T Normalize(T mean, T stdDeviation, T x, bool continuity) {
            static_assert(std::is_floating_point<T>::value, "expect std::is_floating_point<T>::value");

            auto numerator = x - mean;
            if (continuity) {
                // Continuity correction +0.5 because X is already passed with negative deviation
                numerator += 0.5;
            }

            return numerator / stdDeviation;
        }

        template <typename T>
        double Phi(T x) {
            static_assert(std::is_floating_point<T>::value, "expect std::is_floating_point<ValueType>::value");

            return (1.0 + Erf(x / sqrt(2.0))) / 2.0;
        }

        template <typename T>
        double Phi(T mean, T stdDeviation, T x, bool continuity) {
            static_assert(std::is_floating_point<T>::value, "expect std::is_floating_point<ValueType>::value");

            return Phi(Normalize(mean, stdDeviation, x, continuity));
        }

        // Student's t-distribution CDF implementation BEGIN

        //! Continued fraction calculation for incomplete beta-function ratio.
        template <typename ValueType>
        double BetaCF(ValueType a, ValueType b, ValueType x) {
            static const int MAXIT = 200;
            static const ValueType EPS = std::numeric_limits<double>::epsilon();
            static const ValueType FPMIN = std::numeric_limits<double>::min() / EPS;

            /*
             *  See https://dlmf.nist.gov/8.17#v
             *  and https://en.wikipedia.org/wiki/Lentz%27s_algorithm#Algorithm.
             *  Fraction value on each iteration F_i = C_i * D_i * F_(i - 1).
             *  C_i = 1 + numerator_i / C_(i - 1)
             *  D_i = 1 / (1 + numerator_i * D_(i - 1))
             *  Stopping criterion: C_i * D_i ~ 1
             */
            ValueType c = 1.0;
            ValueType d = 1.0 - (a + b) * x / (a + 1);

            if (fabs(d) < FPMIN) {
                d = FPMIN;
            }

            d = 1.0 / d;
            ValueType h = d;

            for (int m = 1; m <= MAXIT; ++m) {
                ValueType numerator = m * (b - m) * x / ((a - 1 + m * 2) * (a + m * 2));
                d = 1.0 + numerator * d;
                c = 1.0 + numerator / c;

                if (fabs(d) < FPMIN) {
                    d = FPMIN;
                }
                if (fabs(c) < FPMIN) {
                    c = FPMIN;
                }

                d = 1.0 / d;
                h *= d * c;
                numerator = -(a + m) * (a + b + m) * x / ((a + m * 2) * (a + 1 + m * 2));
                d = 1.0 + numerator * d;
                c = 1.0 + numerator / c;

                if (fabs(d) < FPMIN) {
                    d = FPMIN;
                }
                if (fabs(c) < FPMIN) {
                    c = FPMIN;
                }

                d = 1.0 / d;
                h *= d * c;

                if (fabs(d * c - 1.0) < EPS) {
                    break;
                }
            }
            return h;
        }

        //! Incomplete beta-function ratio.
        template<typename ValueType>
        double IncompleteBeta(ValueType a, ValueType b, ValueType x) {
            if (x == 0.0 || x == 1.0) {
                return x;
            }

            double logBeta = lgamma(a) + lgamma(b) - lgamma(a + b);
            double front = exp(log(x) * a + log(1.0 - x) * b - logBeta);

            if (x < (a + 1.0) / (a + b + 2.0)) {
                return front * BetaCF(a, b, x) / a;
            } else {
                return 1.0 - front * BetaCF(b, a, 1.0 - x) / b;
            }
        }

        //! t-distribution CDF.
        template<typename ValueType>
        double TCDF(ValueType t, ValueType nu) {
            if (nu <= 0.0) {
                return std::nan("");
            }

            // From Johnson & Kotz & Balakrishnan 28.2 (p.364)
            double x = nu / (nu + t * t);
            double ib = IncompleteBeta(nu / 2.0, 0.5, x);

            if (t > 0) {
                return 1.0 - 0.5 * ib;
            } else {
                return 0.5 * ib;
            }
        }

        // Student's t-distribution CDF implementation END

        // Probit(...) implementation details BEGIN

        //! The derivative of the function Phi.
        template <typename ValueType>
        double DerivativeOfPhi(ValueType x) {
            static_assert(
                std::is_floating_point<ValueType>::value,
                "expect std::is_floating_point<ValueType>::value");

            static const double coefficient = (1.0 / sqrt(2.0 * PI));
            return coefficient * exp(-Sqr(x) / 2.0);
        }

        template <typename ValueType>
        double DerivativeOfPhi(ValueType mean, ValueType stdDeviation, ValueType x) {
            static_assert(
                std::is_floating_point<ValueType>::value,
                "expect std::is_floating_point<ValueType>::value");

            return DerivativeOfPhi(Normalize(mean, stdDeviation, x, false));
        }

        // Probit(...) implementation details END

        // MannWhitney(...) implementation details BEGIN

        //! MWStatistics information used in Mann-Whitney test.
        template <typename ValueType>
        struct MWStatistics {
            ValueType xIndicesSum; /*! The sum of indices of first sample. */
            ValueType yIndicesSum; /*! The sum of indices of second sample. */
            ValueType modifier;    /*! The sum of the squares of the block size of equal values. */
        };

        /*!
           Calculates the MWStatistics from the block of equal values using average indices.
           Adds the result to the MWStatistics object, which passes in the parameters.
        */
        template <typename InputIterator, typename ValueType>
        void GetBlockMWStatistics(
            InputIterator begin, InputIterator end,
            const ValueType commonIndex, MWStatistics<ValueType>& statistics) {
            bool wasXValue = false;
            bool wasYValue = false;
            ValueType size = static_cast<ValueType>(std::distance(begin, end));
            ValueType averageIndex = commonIndex + (size + 1) / 2;

            for (InputIterator it = begin; it != end; ++it) {
                if (!it->second) {
                    wasXValue = true;
                    statistics.xIndicesSum += averageIndex;
                } else {
                    wasYValue = true;
                    statistics.yIndicesSum += averageIndex;
                }
            }

            if (wasXValue && wasYValue) {
                statistics.modifier += size * (Sqr(size) - 1);
            }
        }

        //! Calculates the MWStatistics from the whole range.
        template <typename ValueType, typename InputIterator>
        MWStatistics<ValueType> GetMWStatistics(InputIterator begin, InputIterator end) {
            MWStatistics<ValueType> statistics = MWStatistics<ValueType>();

            InputIterator blockFirst = begin;
            InputIterator blockLast = begin;
            ValueType blockFirstIndex = 0;
            ValueType blockLastIndex = 0;
            for (; blockLast != end; ++blockLast, blockLastIndex += 1) {
                InputIterator next = blockLast;
                ++next;
                if (next == end || !RelativeEqual(blockLast->first, next->first)) {
                    GetBlockMWStatistics(blockFirst, next, blockFirstIndex, statistics);
                    blockFirst = next;
                    blockFirstIndex = blockLastIndex + 1;
                }
            }
            return statistics;
        }

        // MannWhitney(...) implementation details END

        // Wilcoxon(...) implementation details BEGIN

        //! Absolute value comparator for Wilcoxon test.
        template <typename ValueType>
        bool WilcoxonComparator(ValueType a, ValueType b) {
            return fabs(a) < fabs(b);
        }

        //! Wilcoxon test for sorted by absolute value floating numbers.
        template <typename InputIterator>
        NStatistics::TStatTestResult WilcoxonTestWithSign(InputIterator begin, InputIterator end) {
            typedef typename std::iterator_traits<InputIterator>::value_type ValueType;

            const ValueType size = static_cast<ValueType>(std::distance(begin, end));
            ValueType denominator = size * (size + 1) * (2 * size + 1);
            ValueType w = 0;

            InputIterator blockFirst = begin;
            InputIterator blockLast = begin;
            ValueType blockFirstIndex = 0;
            ValueType blockLastIndex = 0;
            for (; blockLast != end; ++blockLast, blockLastIndex += 1) {
                InputIterator next = blockLast;
                ++next;

                if (next == end || !RelativeEqual(*next, *blockFirst)) {
                    const ValueType rank = ((blockFirstIndex + blockLastIndex + 2) / 2.0);

                    for (InputIterator it = blockFirst; it != next; ++it) {
                        if (*it > 0) {
                            w += rank;
                        }
                    }

                    const ValueType blockSize = blockLastIndex - blockFirstIndex + 1;
                    denominator -= blockSize * (blockSize - 1) * (blockSize + 1) * 0.5;
                    blockFirst = next;
                    blockFirstIndex = blockLastIndex + 1;
                }
            }

            if (denominator <= 0) {
                ythrow yexception() << "Incorrect denominator: " << denominator << " <= 0";
            }
            denominator = sqrt(denominator / 24.0);

            const ValueType x = (w - size * (size + 1) / 4.0) / denominator;
            double res = Phi(static_cast<ValueType>(0.0), static_cast<ValueType>(1.0), std::abs(x), false);

            return TStatTestResult((1 - res) * 2, (x > 0) - (x < 0));
        }
        // Wilcoxon(...) implementation details END

        //! Save addition function for Kullback–Leibler divergence
        inline void NonNegativeAdd(double& res, double value, const TStringBuf errorMessage) {
            if (value < 0) {
                ythrow yexception() << errorMessage;
            }
            res += value;
        }
    }

}
