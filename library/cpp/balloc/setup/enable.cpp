#include "enable.h"

#include <util/system/compiler.h>

namespace NAllocSetup {
    // This constant can be overridden on platforms that support weak linkage.
    // See library/cpp/balloc/setup/disable_by_default/disabled.cpp
    extern const bool EnableByDefault Y_WEAK = true;
}
