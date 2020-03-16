#pragma once

#ifdef USE_PYTHON3
#include <contrib/tools/python3/src/Include/context.h>
#else
#error "No <context.h> in Python2"
#endif
