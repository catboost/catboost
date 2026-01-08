#pragma once

#include <util/datetime/base.h>

namespace NYT::NThreading {

////////////////////////////////////////////////////////////////////////////////

#ifdef _linux_

int FutexWait(int* addr, int value, TDuration timeout = TDuration::Max());
int FutexWake(int* addr, int count);

#endif

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NThreading
