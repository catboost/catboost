#pragma once

#ifdef USE_PYTHON3
#include <contrib/tools/python3/src/Include/cpython/compile.h>
#else
#error "No <cpython/compile.h> in Python2"
#endif
