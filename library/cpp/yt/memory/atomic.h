#pragma once

#include <atomic>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

//! A (much) faster version of |fetch_add| for a single-writer case.
template <class T>
T SingleWriterFetchAdd(std::atomic<T>& atomic, T delta);

//! A (much) faster version of |fetch_sub| for a single-writer case.
template <class T>
T SingleWriterFetchSub(std::atomic<T>& atomic, T delta);

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

#define ATOMIC_INL_H_
#include "atomic-inl.h"
#undef ATOMIC_INL_H_
