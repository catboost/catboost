#pragma once

#ifdef USE_PYTHON3
#include <contrib/tools/python3/src/Include/pystrhex.h>
#else
#error "No <pystrhex.h> in Python2"
#endif
