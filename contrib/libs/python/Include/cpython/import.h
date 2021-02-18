#pragma once

#ifdef USE_PYTHON3
#include <contrib/tools/python3/src/Include/cpython/import.h>
#else
#error "No <cpython/import.h> in Python2"
#endif
