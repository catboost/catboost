#pragma once

#include <util/generic/string.h>

#include <string_view>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

//! Compares #lhs with #rhs;
//! returns -1 (if |lhs < rhs|), 0 (if |lhs == rhs|), or 1 (|lhs > rhs|).
template <class T>
int TernaryCompare(const T& lhs, const T& rhs);

//! Same as #TernaryCompare but handles NaN values gracefully
//! (assuming NaNs are larger than any regular number and all NaNs are equal).
//! If |T| is not a floating-point type, #NaNSafeTernaryCompare is equivalent to #TernaryCompare.
template <class T>
int NaNSafeTernaryCompare(const T& lhs, const T& rhs);

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

#define COMPARE_INL_H_
#include "compare-inl.h"
#undef COMPARE_INL_H_
