#pragma once

#ifdef USE_PYTHON3
#include <contrib/tools/python3/src/Include/pyconfig-musl.h>
#else
#error "No <pyconfig-musl.h> in Python2"
#endif
