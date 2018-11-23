#pragma once

#ifdef USE_PYTHON3
#include <contrib/tools/python3/src/Include/namespaceobject.h>
#else
#error "No <namespaceobject.h> in Python2"
#endif
