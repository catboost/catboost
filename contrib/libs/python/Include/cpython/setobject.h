#pragma once

#ifdef USE_PYTHON3
#include <contrib/tools/python3/src/Include/cpython/setobject.h>
#else
#error "No <cpython/setobject.h> in Python2"
#endif
