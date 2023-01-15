#pragma once

#ifdef USE_PYTHON3
#include <contrib/tools/python3/src/Include/accu.h>
#else
#error "No <accu.h> in Python2"
#endif
