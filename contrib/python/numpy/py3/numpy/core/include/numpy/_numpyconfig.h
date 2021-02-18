#include <util/system/platform.h>

#if defined(_windows_)
#   include "_numpyconfig.windows.h"
#elif defined(_darwin_)
#   include "_numpyconfig.darwin.h"
#else
#   include "_numpyconfig.linux.h"
#endif
