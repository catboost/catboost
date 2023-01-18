#pragma once

#ifdef USE_PYTHON3
#include <contrib/tools/python3/src/Include/cpython/cellobject.h>
#else
#error "No <cpython/cellobject.h> in Python2"
#endif
