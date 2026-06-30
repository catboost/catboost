#ifndef NUMERIC_HELPERS_INL_H_
#error "Direct inclusion of this file is not allowed, include numeric_helpers.h"
// For the sake of sane code completion.
#include "numeric_helpers.h"
#endif

#include <cstdlib>
#include <cinttypes>
#include <cmath>
#include <algorithm>

#include <util/system/compiler.h>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

template <class T>
T DivCeil(const T& numerator, const T& denominator)
{
    YT_VERIFY(denominator != 0);
    auto res = std::div(numerator, denominator);
    return res.quot + (res.rem > static_cast<T>(0) ? static_cast<T>(1) : static_cast<T>(0));
}

template <typename T>
T DivRound(const T& numerator, const T& denominator)
{
    auto res = std::div(numerator, denominator);
    return res.quot + (res.rem >= (denominator + 1) / 2 ? static_cast<T>(1) : static_cast<T>(0));
}

template <class T>
T RoundUp(const T& numerator, const T& denominator)
{
    return DivCeil(numerator, denominator) * denominator;
}

template <class T>
T RoundDown(const T& numerator, const T& denominator)
{
    return (numerator / denominator) * denominator;
}

template <class T>
Y_FORCE_INLINE int GetSign(const T& value)
{
    if (value < 0) {
        return -1;
    } else if (value > 0) {
        return +1;
    } else {
        return 0;
    }
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
