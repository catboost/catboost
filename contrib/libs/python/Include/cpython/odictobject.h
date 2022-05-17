#pragma once

#ifdef USE_PYTHON3
#include <contrib/tools/python3/src/Include/cpython/odictobject.h>
#else
#error "No <cpython/odictobject.h> in Python2"
#endif
