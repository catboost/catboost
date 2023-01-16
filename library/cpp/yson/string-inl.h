#pragma once

#ifndef STRING_INL_H_
#error "Direct inclusion of this file is not allowed, include string.h"
// For the sake of sane code completion.
#include "string.h"
#endif

#include <util/str_stl.h>

namespace NYson {

////////////////////////////////////////////////////////////////////////////////

namespace NDetail {

template <typename TLeft, typename TRight>
bool Equals(const TLeft& lhs, const TRight& rhs)
{
    auto lhsNull = !lhs.operator bool();
    auto rhsNull = !rhs.operator bool();
    if (lhsNull != rhsNull) {
        return false;
    }
    if (lhsNull && rhsNull) {
        return true;
    }
    return
        lhs.AsStringBuf() == rhs.AsStringBuf() &&
        lhs.GetType() == rhs.GetType();
}

} // namespace NDetail

inline bool operator == (const TYsonStringBuf& lhs, const TYsonStringBuf& rhs)
{
    return NDetail::Equals(lhs, rhs);
}

inline bool operator != (const TYsonStringBuf& lhs, const TYsonStringBuf& rhs)
{
    return !(lhs == rhs);
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYson

//! A hasher for TYsonStringBuf
template <>
struct THash<NYson::TYsonStringBuf>
{
    size_t operator () (const NYson::TYsonStringBuf& str) const
    {
        return THash<TStringBuf>()(str.AsStringBuf());
    }
};
