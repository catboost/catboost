#pragma once

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

template <class T>
T DivCeil(const T& numerator, const T& denominator);

//! A version of division that is a bit less noisy around the situation
//! when numerator is almost divisible by denominator. Round up if the remainder
//! is at least half of denominator, otherwise round down.
template<class T>
T DivRound(const T& numerator, const T& denominator);

template <class T>
T RoundUp(const T& numerator, const T& denominator);

template <class T>
T RoundDown(const T& numerator, const T& denominator);

template <class T>
int GetSign(const T& value);

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

#define NUMERIC_HELPERS_INL_H_
#include "numeric_helpers-inl.h"
#undef NUMERIC_HELPERS_INL_H_
