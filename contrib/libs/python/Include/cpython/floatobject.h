#pragma once

#ifdef USE_PYTHON3
#include <contrib/tools/python3/src/Include/cpython/floatobject.h>
#else
#error "No <cpython/floatobject.h> in Python2"
#endif
