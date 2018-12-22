#pragma once

#ifdef USE_PYTHON3
#error "No <cStringIO.h> in Python3"
#else
#include <contrib/tools/python/src/Include/cStringIO.h>
#endif
