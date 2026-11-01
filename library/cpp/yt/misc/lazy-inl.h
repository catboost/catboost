#ifndef LAZY_INL_H_
#error "Direct inclusion of this file is not allowed, include lazy.h"
// For the sake of sane code completion.
#include "lazy.h"
#endif

#include <utility>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

template <class T>
decltype(auto) Force(T&& value)
{
    if constexpr (CLazy<T>) {
        // Invoke as const: #TForced is spelled in terms of the const invocation.
        return std::as_const(value).Functor();
    } else {
        return std::forward<T>(value);
    }
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
