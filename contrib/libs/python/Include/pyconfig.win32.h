#pragma once

#ifdef USE_PYTHON3
#error "No <pyconfig.win32.h> in Python3"
#else
#include <contrib/tools/python/src/Include/pyconfig.win32.h>
#endif
