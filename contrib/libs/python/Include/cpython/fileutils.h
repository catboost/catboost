#pragma once

#ifdef USE_PYTHON3
#include <contrib/tools/python3/src/Include/cpython/fileutils.h>
#else
#error "No <cpython/fileutils.h> in Python2"
#endif
