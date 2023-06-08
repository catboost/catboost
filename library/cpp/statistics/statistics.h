#pragma once

#include "detail.h"

#include <util/ysaveload.h>
#include <util/generic/algorithm.h>
#include <util/generic/typetraits.h>
#include <util/generic/utility.h>
#include <util/generic/vector.h>
#include <util/generic/ylimits.h>
#include <util/generic/ymath.h>
#include <util/system/defaults.h>
#include <utility>

#include <cmath>
#include <iterator>

namespace NStatistics {
    //! Normalize x value as if it's from distribution with parameters mean and stdDeviation.
    template <typename ValueType>
    inline ValueType Normalize(ValueType mean, ValueType stdDeviation, ValueType x) {
        NDetail::Normalize(mean, stdDeviation, x);
    }

    //! The inverse function normalize.
    template <typename ValueType>
    ValueType Denormalize(ValueType mean, ValueType stdDeviation, ValueType x) {
        static_assert(std::is_floating_point<ValueType>::value, "expect std::is_floating_point<ValueType>::value");

        return (x * stdDeviation) + mean;
    }

    //! The standard normal cumulative distribution function (phi).
    /*! More details on: http://en.wikipedia.org/wiki/Error_function#Related_functions */
    template <typename ValueType>
    double Phi(ValueType x) {
        return NDetail::Phi(x);
    }

    template <typename ValueType>
    double Phi(ValueType mean, ValueType stdDeviation, ValueType x) {
        return NDetail::Phi(mean, stdDeviation, x);
    }

    //! The inverse function phi. Also known as the normal quantile function.
    /*! More details on: http://en.wikipedia.org/wiki/Error_function#Related_functions */
    template <typename ValueType>
    ValueType Probit(ValueType mean, ValueType stdDeviation, double probability) {
        static_assert(std::is_floating_point<ValueType>::value, "expect std::is_floating_point<ValueType>::value");

        static const double EPS = 1e-7 / static_cast<double>(stdDeviation);
        ValueType x = static_cast<ValueType>(0);

        while (fabs(Phi(x) - probability) > EPS) {
            x -= static_cast<ValueType>((Phi(x) - probability) / NDetail::DerivativeOfPhi(x));
        }
        return Denormalize(mean, stdDeviation, x);
    }

    template <typename ValueType>
    ValueType Probit(double probability) {
        static_assert(std::is_floating_point<ValueType>::value, "expect std::is_floating_point<ValueType>::value");

        return Probit(static_cast<ValueType>(0.0), static_cast<ValueType>(1.0), probability);
    }

    //! Mann-Whitney test.
    /*! More details on: http://en.wikipedia.org/wiki/Mann–Whitney_U */
    template <typename InputIterator1, typename InputIterator2>
    TStatTestResult MannWhitneyWithSign(InputIterator1 xBegin, InputIterator1 xEnd, InputIterator2 yBegin, InputIterator2 yEnd) {
        const size_t MINIMUM_NUMBER_ELEMENTS_NORMAL_APPROXIMATION = 20;

        typedef typename std::iterator_traits<InputIterator1>::value_type ValueType;
        typedef typename std::iterator_traits<InputIterator2>::value_type AnotherValueType;
        static_assert((std::is_same<ValueType, AnotherValueType>::value), "expect (std::is_same<ValueType, AnotherValueType>::value)");
        static_assert(std::is_floating_point<ValueType>::value, "expect std::is_floating_point<ValueType>::value");

        typedef TVector<std::pair<ValueType, bool>> TMWVector;

        ValueType xSize = static_cast<ValueType>(std::distance(xBegin, xEnd));
        ValueType ySize = static_cast<ValueType>(std::distance(yBegin, yEnd));

        if (xSize < MINIMUM_NUMBER_ELEMENTS_NORMAL_APPROXIMATION || ySize < MINIMUM_NUMBER_ELEMENTS_NORMAL_APPROXIMATION) {
            return TStatTestResult(static_cast<ValueType>(1), 0);
        }

        TMWVector xy;
        for (InputIterator1 it = xBegin; it != xEnd; ++it) {
            xy.push_back(std::make_pair(*it, false));
        }
        for (InputIterator2 it = yBegin; it != yEnd; ++it) {
            xy.push_back(std::make_pair(*it, true));
        }

        Sort(xy.begin(), xy.end());
        NDetail::MWStatistics<ValueType> statistics = NDetail::GetMWStatistics<ValueType>(xy.begin(), xy.end());

        ValueType nx = statistics.xIndicesSum > statistics.yIndicesSum ? xSize : ySize;
        ValueType u = xSize * ySize + nx * (nx + 1) / 2 - Max(statistics.xIndicesSum, statistics.yIndicesSum);
        statistics.modifier /= (xSize + ySize) * (Sqr(xSize + ySize) - 1);
        statistics.modifier = sqrt(1 - statistics.modifier);

        if (statistics.modifier < std::numeric_limits<double>::epsilon()) {
            return TStatTestResult(static_cast<ValueType>(1.), 0);
        }

        const ValueType mean = xSize * ySize / 2.0;
        const ValueType stdDeviation = sqrt(xSize * ySize * (xSize + ySize + 1) / 12);
        double res = Phi(mean, stdDeviation * statistics.modifier, u);
        if (res < 0.5) {
            res = 1 - res;
        }
        ValueType xUStatistic = statistics.xIndicesSum - xSize * (xSize + 1) / 2;
        ValueType yUStatistic = statistics.yIndicesSum - ySize * (ySize + 1) / 2;
        int sign = xUStatistic > yUStatistic ? 1 : xUStatistic < yUStatistic ? -1 : 0;
        return TStatTestResult((1 - res) * 2, sign);
    }

    //! Mann-Whitney test.
    /*! More details on: http://en.wikipedia.org/wiki/Mann–Whitney_U */
    template <typename InputIterator1, typename InputIterator2>
    double MannWhitney(InputIterator1 xBegin, InputIterator1 xEnd, InputIterator2 yBegin, InputIterator2 yEnd) {
        return MannWhitneyWithSign(xBegin, xEnd, yBegin, yEnd).PValue;
    }

    //! Wilcoxon test for two samples.
    template <typename InputIterator1, typename InputIterator2>
    TStatTestResult WilcoxonWithSign(InputIterator1 xBegin, InputIterator1 xEnd, InputIterator2 yBegin, InputIterator2 yEnd) {
        typedef typename std::iterator_traits<InputIterator1>::value_type ValueType;
        typedef typename std::iterator_traits<InputIterator2>::value_type AnotherValueType;
        static_assert((std::is_same<ValueType, AnotherValueType>::value), "expect (std::is_same<ValueType, AnotherValueType>::value)");
        static_assert(std::is_floating_point<ValueType>::value, "expect std::is_floating_point<ValueType>::value");

        if (std::distance(xBegin, xEnd) != std::distance(yBegin, yEnd) || xBegin == xEnd) {
            return TStatTestResult(0.5, 0);
        }

        TVector<ValueType> v;
        for (; xBegin != xEnd; ++xBegin, ++yBegin) {
            if (!NDetail::RelativeEqual(*xBegin, *yBegin)) {
                v.push_back(*xBegin - *yBegin);
            }
        }

        Sort(v.begin(), v.end(), NDetail::WilcoxonComparator<ValueType>);
        return NDetail::WilcoxonTestWithSign(v.begin(), v.end());
    }

    //! Wilcoxon test for two samples.
    template <typename InputIterator1, typename InputIterator2>
    double Wilcoxon(InputIterator1 xBegin, InputIterator1 xEnd, InputIterator2 yBegin, InputIterator2 yEnd) {
        return WilcoxonWithSign(xBegin, xEnd, yBegin, yEnd).PValue;
    }

    //! Wilcoxon test for the difference between two samples.
    template <typename InputIterator>
    TStatTestResult WilcoxonWithSign(InputIterator begin, InputIterator end) {
        typedef typename std::iterator_traits<InputIterator>::value_type ValueType;
        static_assert(std::is_floating_point<ValueType>::value, "expect std::is_floating_point<ValueType>::value");

        if (begin == end) {
            return TStatTestResult(0.5, 0);
        }

        TVector<ValueType> v;
        for (InputIterator it = begin; it != end; ++it) {
            if (*it != 0) {
                v.push_back(*it);
            }
        }

        if (v.empty()) {
            return TStatTestResult(0.5, 0);
        }

        Sort(v.begin(), v.end(), NDetail::WilcoxonComparator<ValueType>);
        return NDetail::WilcoxonTestWithSign(v.begin(), v.end());
    }

    //! Wilcoxon test for the difference between two samples.
    template <typename InputIterator>
    double Wilcoxon(InputIterator begin, InputIterator end) {
        return WilcoxonWithSign(begin, end).PValue;
    }

    //! Average of sample.
    template <typename InputIterator>
    typename std::iterator_traits<InputIterator>::value_type
    Average(InputIterator begin, InputIterator end) {
        typedef typename std::iterator_traits<InputIterator>::value_type ValueType;
        static_assert(std::is_floating_point<ValueType>::value, "expect std::is_floating_point<ValueType>::value");

        ValueType sum = static_cast<ValueType>(0);

        for (InputIterator it = begin; it != end; ++it) {
            sum += *it;
        }
        return sum / static_cast<ValueType>(std::distance(begin, end));
    }

    //! Interactive Welford's mean and sample standard deviation computation.
    /*!
        More details on: http://www.johndcook.com/skewness_kurtosis.html

        You might use types different from 'float', 'double', 'long double', for example
        TKahanAccumulator<T> from library/cpp/accurate_accumulate.
        If TStoreFloatType is not one of 'float', 'double', 'long double' it must have
        TStoreFloatType::TValueType.
    */
    template <typename TStoreFloatType, typename TCounterType = ui32>
    class TStatisticsCalculator {
        using TInnerValueType = typename NDetail::TTypeTraits<std::is_arithmetic<TStoreFloatType>::value, TStoreFloatType>::TResult;
        static_assert(std::is_floating_point<TInnerValueType>::value, "expect std::is_floating_point<TInnerValueType>::value");

    public:
        TStatisticsCalculator() {
            Clear();
        }

        template <typename TValueType>
        TStatisticsCalculator(const TCounterType count, const TValueType mean,
                              const TValueType squaredDeviationsSum)
            : Count_(count)
            , M1_(mean)
            , M2_(squaredDeviationsSum)
        {
        }

        void Clear() {
            Count_ = 0;
            M1_ = TStoreFloatType{};
            M2_ = TStoreFloatType{};
        }

        template <typename TValueType>
        void Push(const TValueType value) {
            *this += TStatisticsCalculator<TStoreFloatType, TCounterType>(
                TCounterType(1), TStoreFloatType(value), TStoreFloatType());
        }

        template <typename TValueType>
        void Remove(const TValueType value) {
            *this -= TStatisticsCalculator<TStoreFloatType, TCounterType>(
                TCounterType(1), TStoreFloatType(value), TStoreFloatType());
        }

        inline TCounterType Count() const {
            return Count_;
        }

        inline TStoreFloatType Mean() const {
            return M1_;
        }

        inline TStoreFloatType Variance() const {
            return (Count_ > 1) ? TStoreFloatType(TInnerValueType(M2_) / (Count_ - 1))
                                : TStoreFloatType();
        }

        inline TStoreFloatType StandardDeviation() const {
            return TStoreFloatType(sqrt(TInnerValueType(Variance())));
        }

        inline TStoreFloatType SquaredDeviationsSum() const {
            return M2_;
        }

        template <typename TOtherStoreFloatType, typename TOtherCounterType>
        TStatisticsCalculator<TStoreFloatType, TCounterType>&
        operator+=(const TStatisticsCalculator<TOtherStoreFloatType, TOtherCounterType>& rhs) {
            const TCounterType newCount = Count_ + rhs.Count();
            if (Y_UNLIKELY(0 == newCount)) {
                return *this;
            }

            const TInnerValueType delta = rhs.Mean() - M1_;
            M1_ += delta / newCount * rhs.Count();
            M2_ += rhs.SquaredDeviationsSum() + TStoreFloatType(delta * delta / newCount * Count_ * rhs.Count());

            Count_ = newCount;

            return *this;
        }

        template <typename TOtherStoreFloatType, typename TOtherCounterType>
        TStatisticsCalculator<TStoreFloatType, TCounterType>&
        operator-=(const TStatisticsCalculator<TOtherStoreFloatType, TOtherCounterType>& rhs) {
            Y_ASSERT(Count_ >= rhs.Count());

            const TCounterType newCount = Count_ - rhs.Count();
            if (Y_UNLIKELY(0 == newCount)) {
                Clear();
                return *this;
            }

            const TInnerValueType delta = rhs.Mean() - M1_;
            M1_ -= delta / newCount * rhs.Count();
            M2_ -= rhs.SquaredDeviationsSum() + TStoreFloatType(delta * delta / newCount * Count_ * rhs.Count());

            Count_ = newCount;

            return *this;
        }

        Y_SAVELOAD_DEFINE(Count_, M1_, M2_);

    private:
        TCounterType Count_;

        TStoreFloatType M1_; // Mean: \sum_{i=0}^n (x_i - mean)
        TStoreFloatType M2_; // Sum of squared deviations: \sum_{i=0}^n {(x_i - mean)}^2
    };

    template <typename ValueType,
              typename TCounterType,
              typename TOtherStoreFloatType,
              typename TOtherCounterType>
    inline const TStatisticsCalculator<ValueType, TCounterType>
    operator+(TStatisticsCalculator<ValueType, TCounterType> lhs,
              const TStatisticsCalculator<TOtherStoreFloatType, TOtherCounterType>& rhs) {
        return lhs += rhs;
    }

    template <typename ValueType,
              typename TCounterType,
              typename TOtherStoreFloatType,
              typename TOtherCounterType>
    inline const TStatisticsCalculator<ValueType, TCounterType>
    operator-(TStatisticsCalculator<ValueType, TCounterType> lhs,
              const TStatisticsCalculator<TOtherStoreFloatType, TOtherCounterType>& rhs) {
        return lhs -= rhs;
    }

    template <typename T>
    struct TMeanStd {
        T Mean;
        T Std;
    };

    //! Welford's mean and sample standard deviation computation.
    /*! More details on: http://www.johndcook.com/standard_deviation.html */
    template <typename InputIterator>
    TMeanStd<typename std::iterator_traits<InputIterator>::value_type>
    MeanAndStandardDeviation(InputIterator begin, InputIterator end) {
        using ValueType = typename std::iterator_traits<InputIterator>::value_type;

        TStatisticsCalculator<ValueType> calculator;
        for (InputIterator iter = begin; iter != end; ++iter)
            calculator.Push(*iter);

        return {calculator.Mean(), calculator.StandardDeviation()};
    }

    //! Student's t-test.
    /*! More details on: http://en.wikipedia.org/wiki/Student's_t-test */
    template <typename ValueType>
    double TTest(ValueType diffValue, ValueType diffPrecision, const bool isTailed = false, const bool isLeftTailed = true) {
        static const ValueType EPS = 16 * std::numeric_limits<ValueType>::epsilon();

        if (diffPrecision < EPS) {
            if (std::abs(diffValue) > EPS) {
                return 1.0;
            } else {
                return 0.5;
            }
        }

        if (isTailed) {
            const double res = Phi(static_cast<ValueType>(0.0), diffPrecision, diffValue);
            return isLeftTailed ? res : (1 - res);
        } else {
            const double res = Phi(static_cast<ValueType>(0.0), diffPrecision, std::abs(diffValue));
            return (1 - res) * 2;
        }
    }

    template <typename InputIterator, typename ValueType>
    double TTest(InputIterator begin, InputIterator end, ValueType expectedMean,
                 const bool isTailed = false, const bool isLeftTailed = true) { // = static_cast<ValueType>(0.0)
        typedef typename std::iterator_traits<InputIterator>::value_type AnotherValueType;
        static_assert((std::is_same<ValueType, AnotherValueType>::value), "expect (std::is_same<ValueType, AnotherValueType>::value)");
        static_assert(std::is_floating_point<ValueType>::value, "expect std::is_floating_point<ValueType>::value");

        if (begin == end) {
            return static_cast<ValueType>(0.5);
        }

        const ValueType size = static_cast<ValueType>(std::distance(begin, end));
        const auto meanAndStd = MeanAndStandardDeviation(begin, end);
        return TTest(meanAndStd.Mean - expectedMean, meanAndStd.Std / sqrt(size), isTailed, isLeftTailed);
    }

    template <typename InputIterator1, typename InputIterator2>
    double TTest(InputIterator1 xBegin, InputIterator1 xEnd, InputIterator2 yBegin, InputIterator2 yEnd,
                 const bool isTailed = false, const bool isLeftTailed = true) {
        typedef typename std::iterator_traits<InputIterator1>::value_type ValueType;
        typedef typename std::iterator_traits<InputIterator2>::value_type AnotherValueType;
        static_assert((std::is_same<ValueType, AnotherValueType>::value), "expect (std::is_same<ValueType, AnotherValueType>::value)");
        static_assert(std::is_floating_point<ValueType>::value, "expect std::is_floating_point<ValueType>::value");

        if (xBegin == xEnd || yBegin == yEnd) {
            return static_cast<ValueType>(0.5);
        }

        const ValueType xSize = static_cast<ValueType>(std::distance(xBegin, xEnd));
        const ValueType ySize = static_cast<ValueType>(std::distance(yBegin, yEnd));

        const auto xMeanAndStd = MeanAndStandardDeviation(xBegin, xEnd);
        const auto yMeanAndStd = MeanAndStandardDeviation(yBegin, yEnd);

        const ValueType precision = sqrt(Sqr(xMeanAndStd.Std) / xSize + Sqr(yMeanAndStd.Std) / ySize);

        return TTest(xMeanAndStd.Mean - yMeanAndStd.Mean, precision, isTailed, isLeftTailed);
    }

    //! Kullback–Leibler divergence
    /*! More details on https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence */
    template <typename InputIterator1, typename InputIterator2>
    double KLDivergence(InputIterator1 pBegin, InputIterator1 pEnd, InputIterator2 qBegin, InputIterator2 qEnd) {
        using ValueType = typename std::iterator_traits<InputIterator1>::value_type;
        using AnotherValueType = typename std::iterator_traits<InputIterator2>::value_type;
        static_assert(std::is_convertible<ValueType, double>::value, "P data should be convertible to double");
        static_assert(std::is_convertible<AnotherValueType, double>::value, "Q data should be convertible to double");

        if (pBegin == pEnd || qBegin == qEnd) {
            ythrow yexception() << "Arrays should be non empty";
        }
        auto pIt = pBegin;
        auto qIt = qBegin;
        // formula for non-normalized data: 1/P\sum_{\forall i} p_i * ln(p_i/q_i) + ln(P/Q)
        double pDenominator = 0, qDenominator = 0;
        double divergence = 0;
        for (; pIt != pEnd && qIt != qEnd; ++pIt, ++qIt) {
            NDetail::NonNegativeAdd(pDenominator, *pIt, "Invalid data: p < 0");
            NDetail::NonNegativeAdd(qDenominator, *qIt, "Invalid data: q < 0");
            if (*qIt < std::numeric_limits<double>::epsilon() && *pIt > 0) {
                ythrow yexception() << "Invalid data: q = 0, but p > 0";
            }
            if (*pIt > 0) {
                divergence += *pIt * log(double(*pIt) / *qIt);
            }
        }
        if (pIt != pEnd || qIt != qEnd) {
            ythrow yexception() << "Diffeftent sizes of input data";
        }
        if (pDenominator < std::numeric_limits<double>::epsilon()) {
            ythrow yexception() << "Invalid P denominator";
        }
        if (qDenominator < std::numeric_limits<double>::epsilon()) {
            ythrow yexception() << "Invalid Q denominator";
        }
        divergence /= pDenominator;
        divergence += log(qDenominator / pDenominator);
        return divergence;
    }

    //! Two-sample Kolmogorov–Smirnov statistics for histograms
    /*! More details on https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test#Two-sample_Kolmogorov.E2.80.93Smirnov_test */
    template <typename InputIterator1, typename InputIterator2>
    double KolmogorovSmirnovHistogramStatistics(InputIterator1 pBegin, InputIterator1 pEnd, InputIterator2 qBegin, InputIterator2 qEnd) {
        using ValueType = typename std::iterator_traits<InputIterator1>::value_type;
        using AnotherValueType = typename std::iterator_traits<InputIterator2>::value_type;
        static_assert(std::is_convertible<ValueType, double>::value, "P data should be convertible to double");
        static_assert(std::is_convertible<AnotherValueType, double>::value, "Q data should be convertible to double");

        Y_ENSURE(pBegin != pEnd && qBegin != qEnd, "Arrays should be non empty");
        auto pIt = pBegin;
        auto qIt = qBegin;
        double pDenominator = 0, qDenominator = 0;
        for (; pIt != pEnd && qIt != qEnd; ++pIt, ++qIt) {
            NDetail::NonNegativeAdd(pDenominator, *pIt, "Invalid data: p < 0");
            NDetail::NonNegativeAdd(qDenominator, *qIt, "Invalid data: q < 0");
        }
        Y_ENSURE(pIt == pEnd && qIt == qEnd, "Different sizes of input data");
        Y_ENSURE(pDenominator > 0, "Invalid P denominator");
        Y_ENSURE(qDenominator > 0, "Invalid Q denominator");
        double delta = 0;
        double res = 0;
        for (pIt = pBegin, qIt = qBegin; pIt != pEnd; ++pIt, ++qIt) {
            delta += *pIt / pDenominator - *qIt / qDenominator;
            res = std::max(res, abs(delta));
        }
        return res;
    }

}
