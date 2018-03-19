#pragma once

#include <util/datetime/base.h>

namespace NChromiumTrace {
    TInstant GetThreadCPUTime();
    TInstant GetWallTime();

} // namespace NChromiumTrace
