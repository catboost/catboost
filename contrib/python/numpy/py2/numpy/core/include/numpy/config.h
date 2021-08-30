#include <util/system/platform.h>

#if defined(_windows_)
#   include "config.windows.h"
#elif defined(_darwin_)
#   include "config.darwin.h"
#else
#   include "config.linux.h"
#endif
