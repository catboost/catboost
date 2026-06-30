#include "assert.h"

#include <util/system/yassert.h>
#include <util/system/compiler.h>

namespace NYT::NDetail {

////////////////////////////////////////////////////////////////////////////////

Y_WEAK void AssertTrapImpl(
    TStringBuf trapType,
    TStringBuf expr,
    TStringBuf description,
    TStringBuf file,
    int line,
    TStringBuf function)
{
    // Map to Arcadia assert, poorly...
    ::NPrivate::Panic(
        ::NPrivate::TStaticBuf(file.data(), file.length()),
        line,
        function.data(),
        expr.data(),
        "%s: %.*s",
        trapType.data(),
        static_cast<int>(std::min<ui64>(description.length(), std::numeric_limits<int>::max())),
        description.data());
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NDetail
