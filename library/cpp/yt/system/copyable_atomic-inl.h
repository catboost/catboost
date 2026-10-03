#ifndef COPYABLE_ATOMIC_INL_H_
#error "Direct inclusion of this file is not allowed, include copyable_atomic.h"
// For the sake of sane code completion.
#include "copyable_atomic.h"
#endif

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

template <class T>
TCopyableAtomic<T>::TCopyableAtomic(const TCopyableAtomic& other) noexcept
    : std::atomic<T>(other.load())
{ }

template <class T>
TCopyableAtomic<T>::TCopyableAtomic(TCopyableAtomic&& other) noexcept
    : std::atomic<T>(other.load())
{ }

template <class T>
TCopyableAtomic<T>& TCopyableAtomic<T>::operator=(const TCopyableAtomic& other) noexcept
{
    this->store(other.load());
    return *this;
}

template <class T>
TCopyableAtomic<T>& TCopyableAtomic<T>::operator=(TCopyableAtomic&& other) noexcept
{
    this->store(other.load());
    return *this;
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
