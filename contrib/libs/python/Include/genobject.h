#pragma once

#ifdef USE_PYTHON3
#error "No <genobject.h> in Python3"
#else
#include <contrib/tools/python/src/Include/genobject.h>
#endif
