#pragma once

#ifdef USE_PYTHON3
#include <contrib/tools/python3/src/Include/cpython/longintrepr.h>
#else
#error "No <cpython/longintrepr.h> in Python2"
#endif
