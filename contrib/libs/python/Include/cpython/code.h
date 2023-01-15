#pragma once

#ifdef USE_PYTHON3
#include <contrib/tools/python3/src/Include/cpython/code.h>
#else
#error "No <cpython/code.h> in Python2"
#endif
