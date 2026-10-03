#pragma once

#include <atomic>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

//! A drop-in replacement for |std::atomic| that can be copied and moved,
//! which makes it usable as a member of a copyable class.
/*!
 *  Copying is not atomic: the source is loaded and the result is used to
 *  initialize (or is stored into) the target.
 */
template <class T>
struct TCopyableAtomic
    : public std::atomic<T>
{
    using std::atomic<T>::atomic;
    using std::atomic<T>::operator=;

    TCopyableAtomic() = default;

    TCopyableAtomic(const TCopyableAtomic& other) noexcept;
    TCopyableAtomic(TCopyableAtomic&& other) noexcept;

    TCopyableAtomic& operator=(const TCopyableAtomic& other) noexcept;
    TCopyableAtomic& operator=(TCopyableAtomic&& other) noexcept;
};

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

#define COPYABLE_ATOMIC_INL_H_
#include "copyable_atomic-inl.h"
#undef COPYABLE_ATOMIC_INL_H_
