#pragma once

#include <util/ysaveload.h>
#include <util/generic/vector.h>
#include <util/system/yassert.h>

//! See more details here http://en.wikipedia.org/wiki/Kahan_summation_algorithm
template <typename TAccumulateType>
class TKahanAccumulator {
public:
    using TValueType = TAccumulateType;

    template <typename TFloatType>
    explicit TKahanAccumulator(const TFloatType x)
        : Sum_(x)
        , Compensation_()
    {
    }

    TKahanAccumulator()
        : Sum_()
        , Compensation_()
    {
    }

    template <typename TFloatType>
    TKahanAccumulator& operator=(const TFloatType& rhs) {
        Sum_ = TValueType(rhs);
        Compensation_ = TValueType();
        return *this;
    }

    TValueType Get() const {
        return Sum_ + Compensation_;
    }

    template <typename TFloatType>
    inline operator TFloatType() const {
        return Get();
    }

    template <typename TFloatType>
    inline bool operator<(const TKahanAccumulator<TFloatType>& other) const {
        return Get() < other.Get();
    }

    template <typename TFloatType>
    inline bool operator<=(const TKahanAccumulator<TFloatType>& other) const {
        return !(other < *this);
    }

    template <typename TFloatType>
    inline bool operator>(const TKahanAccumulator<TFloatType>& other) const {
        return other < *this;
    }

    template <typename TFloatType>
    inline bool operator>=(const TKahanAccumulator<TFloatType>& other) const {
        return !(*this < other);
    }

    template <typename TFloatType>
    inline TKahanAccumulator& operator+=(const TFloatType x) {
        const TValueType y = TValueType(x) - Compensation_;
        const TValueType t = Sum_ + y;
        Compensation_ = (t - Sum_) - y;
        Sum_ = t;
        return *this;
    }

    template <typename TFloatType>
    inline TKahanAccumulator& operator-=(const TFloatType x) {
        return *this += -TValueType(x);
    }

    template <typename TFloatType>
    inline TKahanAccumulator& operator*=(const TFloatType x) {
        return *this = TValueType(*this) * TValueType(x);
    }

    template <typename TFloatType>
    inline TKahanAccumulator& operator/=(const TFloatType x) {
        return *this = TValueType(*this) / TValueType(x);
    }

    Y_SAVELOAD_DEFINE(Sum_, Compensation_);

private:
    TValueType Sum_;
    TValueType Compensation_;
};

template <typename TAccumulateType, typename TFloatType>
inline const TKahanAccumulator<TAccumulateType>
operator+(TKahanAccumulator<TAccumulateType> lhs, const TFloatType rhs) {
    return lhs += rhs;
}

template <typename TAccumulateType, typename TFloatType>
inline const TKahanAccumulator<TAccumulateType>
operator-(TKahanAccumulator<TAccumulateType> lhs, const TFloatType rhs) {
    return lhs -= rhs;
}

template <typename TAccumulateType, typename TFloatType>
inline const TKahanAccumulator<TAccumulateType>
operator*(TKahanAccumulator<TAccumulateType> lhs, const TFloatType rhs) {
    return lhs *= rhs;
}

template <typename TAccumulateType, typename TFloatType>
inline const TKahanAccumulator<TAccumulateType>
operator/(TKahanAccumulator<TAccumulateType> lhs, const TFloatType rhs) {
    return lhs /= rhs;
}

template <typename TAccumulatorType, typename It>
static inline TAccumulatorType TypedFastAccumulate(It begin, It end) {
    TAccumulatorType accumulator = TAccumulatorType();

    for (; begin + 15 < end; begin += 16) {
        accumulator += *(begin + 0) +
                       *(begin + 1) +
                       *(begin + 2) +
                       *(begin + 3) +
                       *(begin + 4) +
                       *(begin + 5) +
                       *(begin + 6) +
                       *(begin + 7) +
                       *(begin + 8) +
                       *(begin + 9) +
                       *(begin + 10) +
                       *(begin + 11) +
                       *(begin + 12) +
                       *(begin + 13) +
                       *(begin + 14) +
                       *(begin + 15);
    }
    for (; begin != end; ++begin) {
        accumulator += *begin;
    }

    return accumulator;
}

template <class TOperation, typename TAccumulatorType, typename It1, typename It2>
static inline TAccumulatorType TypedFastInnerOperation(It1 begin1, It1 end1, It2 begin2) {
    TAccumulatorType accumulator = TAccumulatorType();

    const TOperation op;
    for (; begin1 + 15 < end1; begin1 += 16, begin2 += 16) {
        accumulator += op(*(begin1 + 0), *(begin2 + 0)) +
                       op(*(begin1 + 1), *(begin2 + 1)) +
                       op(*(begin1 + 2), *(begin2 + 2)) +
                       op(*(begin1 + 3), *(begin2 + 3)) +
                       op(*(begin1 + 4), *(begin2 + 4)) +
                       op(*(begin1 + 5), *(begin2 + 5)) +
                       op(*(begin1 + 6), *(begin2 + 6)) +
                       op(*(begin1 + 7), *(begin2 + 7)) +
                       op(*(begin1 + 8), *(begin2 + 8)) +
                       op(*(begin1 + 9), *(begin2 + 9)) +
                       op(*(begin1 + 10), *(begin2 + 10)) +
                       op(*(begin1 + 11), *(begin2 + 11)) +
                       op(*(begin1 + 12), *(begin2 + 12)) +
                       op(*(begin1 + 13), *(begin2 + 13)) +
                       op(*(begin1 + 14), *(begin2 + 14)) +
                       op(*(begin1 + 15), *(begin2 + 15));
    }
    for (; begin1 != end1; ++begin1, ++begin2) {
        accumulator += op(*begin1, *begin2);
    }

    return accumulator;
}

template <typename TAccumulatorType, typename It1, typename It2>
static inline TAccumulatorType TypedFastInnerProduct(It1 begin1, It1 end1, It2 begin2) {
    return TypedFastInnerOperation<std::multiplies<>, TAccumulatorType>(begin1, end1, begin2);
}

template <typename It>
static inline double FastAccumulate(It begin, It end) {
    return TypedFastAccumulate<double>(begin, end);
}

template <typename T>
static inline double FastAccumulate(const TVector<T>& sequence) {
    return FastAccumulate(sequence.begin(), sequence.end());
}

template <typename It>
static inline double FastKahanAccumulate(It begin, It end) {
    return TypedFastAccumulate<TKahanAccumulator<double>>(begin, end);
}

template <typename T>
static inline double FastKahanAccumulate(const TVector<T>& sequence) {
    return FastKahanAccumulate(sequence.begin(), sequence.end());
}

template <typename It1, typename It2>
static inline double FastInnerProduct(It1 begin1, It1 end1, It2 begin2) {
    return TypedFastInnerProduct<double>(begin1, end1, begin2);
}

template <typename T>
static inline double FastInnerProduct(const TVector<T>& lhs, const TVector<T>& rhs) {
    Y_ASSERT(lhs.size() == rhs.size());
    return FastInnerProduct(lhs.begin(), lhs.end(), rhs.begin());
}

template <typename It1, typename It2>
static inline double FastKahanInnerProduct(It1 begin1, It1 end1, It2 begin2) {
    return TypedFastInnerProduct<TKahanAccumulator<double>>(begin1, end1, begin2);
}

template <typename T>
static inline double FastKahanInnerProduct(const TVector<T>& lhs, const TVector<T>& rhs) {
    Y_ASSERT(lhs.size() == rhs.size());
    return FastKahanInnerProduct(lhs.begin(), lhs.end(), rhs.begin());
}
