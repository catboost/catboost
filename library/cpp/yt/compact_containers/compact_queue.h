#pragma once

#include "compact_vector.h"

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

//! A queue optimized for storing elements inline
//! and with little memory overhead. See TCompactVector.
template <class T, size_t N>
class TCompactQueue
{
public:
    void Push(T value);
    T Pop();

    const T& Front() const;

    size_t Size() const;
    size_t Capacity() const;

    bool Empty() const;

private:
    TCompactVector<T, N> Queue_ = TCompactVector<T, N>(N);
    size_t FrontIndex_ = 0;
    size_t Size_ = 0;
};

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

#define COMPACT_QUEUE_INL_H_
#include "compact_queue-inl.h"
#undef COMPACT_QUEUE_INL_H_
