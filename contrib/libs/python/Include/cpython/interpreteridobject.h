#pragma once

#ifdef USE_PYTHON3
#include <contrib/tools/python3/src/Include/cpython/interpreteridobject.h>
#else
#error "No <cpython/interpreteridobject.h> in Python2"
#endif
