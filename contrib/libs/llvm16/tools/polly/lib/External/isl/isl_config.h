#pragma once

#if defined(__APPLE__)
#   include "isl_config-osx.h"
#elif defined(_MSC_VER)
#   include "isl_config-win.h"
#else
#   include "isl_config-linux.h"
#endif
