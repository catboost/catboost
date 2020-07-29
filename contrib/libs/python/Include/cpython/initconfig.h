#pragma once

#ifdef USE_PYTHON3
#include <contrib/tools/python3/src/Include/cpython/initconfig.h>
#else
#error "No <cpython/initconfig.h> in Python2"
#endif
