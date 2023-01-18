#pragma once

#ifdef USE_PYTHON3
#include <contrib/tools/python3/src/Include/cpython/context.h>
#else
#error "No <cpython/context.h> in Python2"
#endif
