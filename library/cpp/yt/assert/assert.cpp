#include "assert.h"

#include <util/system/yassert.h>
#include <util/system/compiler.h>

namespace NYT::NDetail {

////////////////////////////////////////////////////////////////////////////////

Y_WEAK void AssertTrapImpl(
    TStringBuf trapType,
    TStringBuf expr,
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
        "%s",
        trapType.data());
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NDetail
