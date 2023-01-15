#pragma once

#ifdef USE_PYTHON3
#include <contrib/tools/python3/src/Include/cpython/bytesobject.h>
#else
#error "No <cpython/bytesobject.h> in Python2"
#endif
