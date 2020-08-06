#pragma once

#ifdef USE_PYTHON3
#include <contrib/tools/python3/src/Include/picklebufobject.h>
#else
#error "No <picklebufobject.h> in Python2"
#endif
