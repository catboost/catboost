#ifndef LEAKY_SINGLETON_INL_H_
#error "Direct inclusion of this file is not allowed, include leaky_singleton.h"
// For the sake of sane code completion.
#include "leaky_singleton.h"
#endif

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

template <class T>
TLeakyStorage<T>::TLeakyStorage()
{
    new (Get()) T();
}

template <class T>
T* TLeakyStorage<T>::Get()
{
    return reinterpret_cast<T*>(Buffer_);
}

////////////////////////////////////////////////////////////////////////////////

template <class T>
T* LeakySingleton()
{
    static TLeakyStorage<T> Storage;
    return Storage.Get();
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
