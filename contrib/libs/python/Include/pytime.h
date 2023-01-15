#pragma once

#ifdef USE_PYTHON3
#include <contrib/tools/python3/src/Include/pytime.h>
#else
#error "No <pytime.h> in Python2"
#endif
