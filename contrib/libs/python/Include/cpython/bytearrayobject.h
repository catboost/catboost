#pragma once

#ifdef USE_PYTHON3
#include <contrib/tools/python3/src/Include/cpython/bytearrayobject.h>
#else
#error "No <cpython/bytearrayobject.h> in Python2"
#endif
