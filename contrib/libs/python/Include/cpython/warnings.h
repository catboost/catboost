#pragma once

#ifdef USE_PYTHON3
#include <contrib/tools/python3/src/Include/cpython/warnings.h>
#else
#error "No <cpython/warnings.h> in Python2"
#endif
