#pragma once

#ifdef USE_PYTHON3
#include <contrib/tools/python3/src/Include/cpython/complexobject.h>
#else
#error "No <cpython/complexobject.h> in Python2"
#endif
