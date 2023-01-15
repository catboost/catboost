#pragma once

#ifdef USE_PYTHON3
#error "No <pyconfig.freebsd.h> in Python3"
#else
#include <contrib/tools/python/src/Include/pyconfig.freebsd.h>
#endif
