#pragma once

#ifdef USE_PYTHON3
#error "No <stringobject.h> in Python3"
#else
#include <contrib/tools/python/src/Include/stringobject.h>
#endif
