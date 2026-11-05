#ifndef ATOMIC_INL_H_
#error "Direct inclusion of this file is not allowed, include atomic.h"
// For the sake of sane code completion.
#include "atomic.h"
#endif

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

template <class T>
T SingleWriterFetchAdd(std::atomic<T>& atomic, T delta)
{
    auto oldValue = atomic.load(std::memory_order::relaxed);
    atomic.store(oldValue + delta, std::memory_order::relaxed);
    return oldValue;
}

template <class T>
T SingleWriterFetchSub(std::atomic<T>& atomic, T delta)
{
    auto oldValue = atomic.load(std::memory_order::relaxed);
    atomic.store(oldValue - delta, std::memory_order::relaxed);
    return oldValue;
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
