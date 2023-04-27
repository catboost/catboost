#pragma once

#if defined(__ANDROID__)
#   include "ares_config-android.h"
#elif defined(__APPLE__)
#   include "ares_config-osx.h"
#else
#   include "ares_config-linux.h"
#endif
