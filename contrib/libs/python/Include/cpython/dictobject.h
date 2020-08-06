#pragma once

#ifdef USE_PYTHON3
#include <contrib/tools/python3/src/Include/cpython/dictobject.h>
#else
#error "No <cpython/dictobject.h> in Python2"
#endif
