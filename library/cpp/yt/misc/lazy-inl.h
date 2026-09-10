#ifndef LAZY_INL_H_
#error "Direct inclusion of this file is not allowed, include lazy.h"
// For the sake of sane code completion.
#include "lazy.h"
#endif

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

template <class T>
decltype(auto) Unlazy(T&& value)
{
    if constexpr (CLazy<T>) {
        return value.Functor();
    } else {
        return std::forward<T>(value);
    }
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
