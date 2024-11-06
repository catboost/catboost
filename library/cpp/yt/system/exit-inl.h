#ifndef EXIT_INL_H_
#error "Direct inclusion of this file is not allowed, include exit.h"
// For the sake of sane code completion.
#include "exit.h"
#endif

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

template <class E>
    requires std::is_enum_v<E>
[[noreturn]] void AbortProcessSilently(E exitCode)
{
    AbortProcessSilently(ToUnderlying(exitCode));
}

template <class E>
    requires std::is_enum_v<E>
[[noreturn]] void AbortProcessDramatically(
    E exitCode,
    TStringBuf message)
{
    AbortProcessDramatically(
        ToUnderlying(exitCode),
        TEnumTraits<E>::FindLiteralByValue(exitCode).value_or("<unknown>"),
        message);
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
