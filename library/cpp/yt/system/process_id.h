#pragma once

#include <util/system/getpid.h>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

constexpr TProcessId InvalidProcessId = TProcessId(-1);

//! Returns the OS process id (|getpid|).
//! The value is cached process-wide, so only the first call hits the kernel.
//! The cache is reset in the child after |fork|.
TProcessId GetProcessId();

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

#define PROCESS_ID_INL_H_
#include "process_id-inl.h"
#undef PROCESS_ID_INL_H_
