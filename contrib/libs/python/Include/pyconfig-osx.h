#pragma once

#ifdef USE_PYTHON3
#include <contrib/tools/python3/src/Include/pyconfig-osx.h>
#else
#error "No <pyconfig-osx.h> in Python2"
#endif
