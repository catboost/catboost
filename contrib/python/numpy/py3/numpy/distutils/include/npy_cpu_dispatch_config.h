#include <util/system/platform.h>

#if defined(_x86_64_)
#   include "npy_cpu_dispatch_config.x86_64.h"
#elif defined(_arm64_)
#   include "npy_cpu_dispatch_config.aarch64.h"
#endif
