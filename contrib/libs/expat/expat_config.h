#pragma once

#if defined(__IOS__)
#   include "expat_config-ios.h"
#elif defined(_MSC_VER)
#   include "expat_config-win.h"
#else
#   include "expat_config-linux.h"
#endif
