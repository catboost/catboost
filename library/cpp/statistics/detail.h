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
        T Normalize(T mean, T stdDeviation, T x) {
            static_assert(std::is_floating_point<T>::value, "expect std::is_floating_point<T>::value");

            return (x - mean) / stdDeviation;
        }

        template <typename T>
        double Phi(T x) {
            static_assert(std::is_floating_point<T>::value, "expect std::is_floating_point<ValueType>::value");

            return (1.0 + Erf(x / sqrt(2.0))) / 2.0;
        }

        template <typename T>
        double Phi(T mean, T stdDeviation, T x) {
            static_assert(std::is_floating_point<T>::value, "expect std::is_floating_point<ValueType>::value");

            return Phi(Normalize(mean, stdDeviation, x));
        }

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

            return DerivativeOfPhi(Normalize(mean, stdDeviation, x));
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
            double res = Phi(static_cast<ValueType>(0.0), static_cast<ValueType>(1.0), std::abs(x));

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
