#ifndef LEAKY_SINGLETON_INL_H_
#error "Direct inclusion of this file is not allowed, include leaky_singleton.h"
// For the sake of sane code completion.
#include "leaky_singleton.h"
#endif

#ifdef _asan_enabled_
#include <sanitizer/lsan_interface.h>
#endif

#include <utility>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

template <class T>
template <class... TArgs>
TLeakyStorage<T>::TLeakyStorage(TArgs&&... args)
{
#ifdef _asan_enabled_
    __lsan_disable();
#endif
    new (Get()) T(std::forward<TArgs>(args)...);
#ifdef _asan_enabled_
    __lsan_enable();
#endif
}

template <class T>
T* TLeakyStorage<T>::Get()
{
    return reinterpret_cast<T*>(Buffer_);
}

////////////////////////////////////////////////////////////////////////////////

template <class T, class... TArgs>
T* LeakySingleton(TArgs&&... args)
{
    static TLeakyStorage<T> Storage(std::forward<TArgs>(args)...);
    return Storage.Get();
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
