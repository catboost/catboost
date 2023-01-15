#pragma once

#ifdef USE_PYTHON3
#include <contrib/tools/python3/src/Include/cpython/pylifecycle.h>
#else
#error "No <cpython/pylifecycle.h> in Python2"
#endif
