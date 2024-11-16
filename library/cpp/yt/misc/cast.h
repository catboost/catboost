#pragma once

#include "enum.h"

#include <library/cpp/yt/exception/exception.h>

#include <optional>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

template <class T, class S>
constexpr bool CanFitSubtype();

template <class T, class S>
constexpr bool IsInIntegralRange(S value);

template <class T, class S>
constexpr std::optional<T> TryCheckedIntegralCast(S value);

template <class T, class S>
T CheckedIntegralCast(S value);

////////////////////////////////////////////////////////////////////////////////

template <class T, class S>
    requires TEnumTraits<T>::IsEnum
constexpr std::optional<T> TryCheckedEnumCast(S value, bool enableUnknown = false);

template <class T, class S>
    requires TEnumTraits<T>::IsEnum
T CheckedEnumCast(S value);

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

#define CAST_INL_H_
#include "cast-inl.h"
#undef CAST_INL_H_
