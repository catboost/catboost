#pragma once

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

template <class T>
class TLeakyStorage
{
public:
    template <class... TArgs>
    explicit TLeakyStorage(TArgs&&... args);

    T* Get();

private:
    alignas(T) char Buffer_[sizeof(T)];
};

////////////////////////////////////////////////////////////////////////////////

#define DECLARE_LEAKY_SINGLETON_FRIEND() \
    template <class T>                   \
    friend class ::NYT::TLeakyStorage;

template <class T, class... TArgs>
T* LeakySingleton(TArgs&&... args);

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

#define LEAKY_SINGLETON_INL_H_
#include "leaky_singleton-inl.h"
#undef LEAKY_SINGLETON_INL_H_
