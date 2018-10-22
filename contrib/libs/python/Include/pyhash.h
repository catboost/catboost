#pragma once

#ifdef USE_PYTHON3
#include <contrib/tools/python3/src/Include/pyhash.h>
#else
#error "No <pyhash.h> in Python2"
#endif
