#pragma once

#ifdef USE_PYTHON3
#include <contrib/tools/python3/src/Include/cpython/objimpl.h>
#else
#error "No <cpython/objimpl.h> in Python2"
#endif
