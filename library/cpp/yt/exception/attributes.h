#pragma once

#include <library/cpp/yt/misc/guid.h>

#include <string>

#include <variant>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

struct TExceptionAttribute
{
    using TKey = std::string;
    using TValue = std::variant<i64, double, bool, std::string>;

    TKey Key;
    TValue Value;
};

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
