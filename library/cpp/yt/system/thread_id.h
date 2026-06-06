#pragma once

#include <util/generic/ylimits.h>

#include <util/system/types.h>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

using TSystemThreadId = size_t;
constexpr TSystemThreadId InvalidSystemThreadId = Max<TSystemThreadId>();
//! Returns the OS thread id (e.g. |gettid| on Linux).
//! The value is cached in TLS, so only the first call per thread hits the kernel.
//! The cache is reset in the child after |fork|.
TSystemThreadId GetSystemThreadId();

using TSequentialThreadId = ui32;
constexpr TSequentialThreadId InvalidSequentialThreadId = Max<TSequentialThreadId>();
TSequentialThreadId GetSequentialThreadId();

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

#define THREAD_ID_INL_H_
#include "thread_id-inl.h"
#undef THREAD_ID_INL_H_
