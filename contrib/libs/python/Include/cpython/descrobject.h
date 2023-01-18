#pragma once

#ifdef USE_PYTHON3
#include <contrib/tools/python3/src/Include/cpython/descrobject.h>
#else
#error "No <cpython/descrobject.h> in Python2"
#endif
