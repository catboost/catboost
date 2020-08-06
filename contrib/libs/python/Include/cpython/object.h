#pragma once

#ifdef USE_PYTHON3
#include <contrib/tools/python3/src/Include/cpython/object.h>
#else
#error "No <cpython/object.h> in Python2"
#endif
