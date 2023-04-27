#pragma once

#if defined(__APPLE__)
#   include "opj_config_private-osx.h"
#elif defined(_MSC_VER)
#   include "opj_config_private-win.h"
#else
#   include "opj_config_private-linux.h"
#endif
