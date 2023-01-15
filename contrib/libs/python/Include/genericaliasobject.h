#pragma once

#ifdef USE_PYTHON3
#include <contrib/tools/python3/src/Include/genericaliasobject.h>
#else
#error "No <genericaliasobject.h> in Python2"
#endif
