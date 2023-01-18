#pragma once

#ifdef USE_PYTHON3
#include <contrib/tools/python3/src/Include/pytypedefs.h>
#else
#error "No <pytypedefs.h> in Python2"
#endif
