#pragma once

#include "intrusive_ptr.h"

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

template <class T>
TIntrusivePtr<T> LeakyRefCountedSingleton();

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

#define LEAKY_REF_COUNTED_SINGLETON_INL_H_
#include "leaky_ref_counted_singleton-inl.h"
#undef LEAKY_REF_COUNTED_SINGLETON_INL_H_
