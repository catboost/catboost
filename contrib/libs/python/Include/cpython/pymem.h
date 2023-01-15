#pragma once

#ifdef USE_PYTHON3
#include <contrib/tools/python3/src/Include/cpython/pymem.h>
#else
#error "No <cpython/pymem.h> in Python2"
#endif
