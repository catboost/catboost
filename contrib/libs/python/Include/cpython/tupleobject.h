#pragma once

#ifdef USE_PYTHON3
#include <contrib/tools/python3/src/Include/cpython/tupleobject.h>
#else
#error "No <cpython/tupleobject.h> in Python2"
#endif
