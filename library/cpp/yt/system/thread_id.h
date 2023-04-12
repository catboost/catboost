#pragma once

#include <util/generic/ylimits.h>

#include <util/system/types.h>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

using TSystemThreadId = size_t;
constexpr TSystemThreadId InvalidSystemThreadId = Max<TSystemThreadId>();
TSystemThreadId GetSystemThreadId();

using TSequentialThreadId = ui32;
constexpr TSequentialThreadId InvalidSequentialThreadId = Max<TSequentialThreadId>();
TSequentialThreadId GetSequentialThreadId();

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NThreading

#define THREAD_ID_INL_H_
#include "thread_id-inl.h"
#undef THREAD_ID_INL_H_
