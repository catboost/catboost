#pragma once

#ifdef USE_PYTHON3
#include <contrib/tools/python3/src/Include/cpython/ceval.h>
#else
#error "No <cpython/ceval.h> in Python2"
#endif
