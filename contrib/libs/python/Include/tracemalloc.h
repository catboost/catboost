#pragma once

#ifdef USE_PYTHON3
#include <contrib/tools/python3/src/Include/tracemalloc.h>
#else
#error "No <tracemalloc.h> in Python2"
#endif
