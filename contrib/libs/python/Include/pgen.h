#pragma once

#ifdef USE_PYTHON3
#error "No <pgen.h> in Python3"
#else
#include <contrib/tools/python/src/Include/pgen.h>
#endif
