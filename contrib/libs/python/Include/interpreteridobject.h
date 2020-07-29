#pragma once

#ifdef USE_PYTHON3
#include <contrib/tools/python3/src/Include/interpreteridobject.h>
#else
#error "No <interpreteridobject.h> in Python2"
#endif
