#pragma once

#ifdef USE_PYTHON3
#include <contrib/tools/python3/src/Include/cpython/classobject.h>
#else
#error "No <cpython/classobject.h> in Python2"
#endif
