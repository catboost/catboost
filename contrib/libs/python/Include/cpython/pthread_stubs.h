#pragma once

#ifdef USE_PYTHON3
#include <contrib/tools/python3/src/Include/cpython/pthread_stubs.h>
#else
#error "No <cpython/pthread_stubs.h> in Python2"
#endif
