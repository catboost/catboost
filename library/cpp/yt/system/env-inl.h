#ifndef ENV_INL_H_
#error "Direct inclusion of this file is not allowed, include env.h"
// For the sake of sane code completion.
#include "env.h"
#endif

#include <util/string/cast.h>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

namespace NDetail {

[[noreturn]] void ThrowFailedToParseEnvValueError(TStringBuf name, TStringBuf value);

} // namespace NDetail

template <class T>
T GetEnvValueOrThrow(TStringBuf name)
{
    auto value = GetEnvValueOrThrow(name);
    T result;
    if (!TryFromString<T>(value, result)) {
        NDetail::ThrowFailedToParseEnvValueError(name, value);
    }
    return result;
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
