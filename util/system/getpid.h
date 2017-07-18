#pragma once

#include "platform.h"
#include "types.h"

#if defined(_win_)
using TProcessId = ui32; // DWORD
#else
using TProcessId = pid_t;
#endif

TProcessId GetPID();
