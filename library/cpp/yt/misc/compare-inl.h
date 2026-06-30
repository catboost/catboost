#ifndef COMPARE_INL_H_
#error "Direct inclusion of this file is not allowed, include compare.h"
// For the sake of sane code completion.
#include "compare.h"
#endif

#include "numeric_helpers.h"

#include <util/generic/string.h>

#include <string>
#include <string_view>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

template <class T>
Y_FORCE_INLINE int TernaryCompare(const T& lhs, const T& rhs)
{
    if (lhs == rhs) {
        return 0;
    } else if (lhs < rhs) {
        return -1;
    } else {
        return +1;
    }
}

//! An optimized version for string types.
template <class T>
    requires
        std::is_same_v<T, TString> ||
        std::is_same_v<T, TStringBuf> ||
        std::is_same_v<T, std::string> ||
        std::is_same_v<T, std::string_view>
Y_FORCE_INLINE int TernaryCompare(const T& lhs, const T& rhs)
{
    return GetSign(std::string_view(lhs).compare(std::string_view(rhs)));
}

template <class T>
Y_FORCE_INLINE int NaNSafeTernaryCompare(const T& lhs, const T& rhs)
{
    return TernaryCompare(lhs, rhs);
}

template <class T>
    requires std::is_floating_point_v<T>
Y_FORCE_INLINE int NaNSafeTernaryCompare(const T& lhs, const T& rhs)
{
    if (lhs < rhs) {
        return -1;
    } else if (lhs > rhs) {
        return +1;
    } else if (std::isnan(lhs)) {
        if (std::isnan(rhs)) {
            return 0;
        } else {
            return +1;
        }
    } else if (std::isnan(rhs)) {
        return -1;
    } else {
        return 0;
    }
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
