#pragma once

#ifdef USE_PYTHON3
#include <contrib/tools/python3/src/Include/cpython/pytime.h>
#else
#error "No <cpython/pytime.h> in Python2"
#endif
