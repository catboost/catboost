#pragma once

#ifdef USE_PYTHON3
#include <contrib/tools/python3/src/Include/pyconfig-linux.h>
#else
#error "No <pyconfig-linux.h> in Python2"
#endif
