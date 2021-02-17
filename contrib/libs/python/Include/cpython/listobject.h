#pragma once

#ifdef USE_PYTHON3
#include <contrib/tools/python3/src/Include/cpython/listobject.h>
#else
#error "No <cpython/listobject.h> in Python2"
#endif
