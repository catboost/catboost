#pragma once

#ifdef USE_PYTHON3
#error "No <pyctype.h> in Python3"
#else
#include <contrib/tools/python/src/Include/pyctype.h>
#endif
