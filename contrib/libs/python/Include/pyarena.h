#pragma once

#ifdef USE_PYTHON3
#error "No <pyarena.h> in Python3"
#else
#include <contrib/tools/python/src/Include/pyarena.h>
#endif
