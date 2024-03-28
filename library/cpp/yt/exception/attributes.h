#pragma once

#include <library/cpp/yt/misc/guid.h>

#include <util/generic/string.h>

#include <variant>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

struct TExceptionAttribute
{
    using TKey = TString;
    using TValue = std::variant<i64, double, bool, TString>;

    template <class T>
    TExceptionAttribute(const TKey& key, const T& value)
        : Key(key)
        , Value(value)
    { }

    TKey Key;
    TValue Value;
};

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
