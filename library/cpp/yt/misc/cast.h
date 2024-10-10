#pragma once

#include <library/cpp/yt/exception/exception.h>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

template <class T, class S>
constexpr bool CanFitSubtype();

template <class T, class S>
constexpr bool IsInIntegralRange(S value);

template <class T, class S>
constexpr bool TryIntegralCast(S value, T* result);

template <class T, class S>
T CheckedIntegralCast(S value);

////////////////////////////////////////////////////////////////////////////////

template <class T, class S>
constexpr bool TryEnumCast(S value, T* result);

template <class T, class S>
T CheckedEnumCast(S value);

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

#define CAST_INL_H_
#include "cast-inl.h"
#undef CAST_INL_H_
