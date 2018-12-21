#pragma once

#ifdef USE_PYTHON3
#include <contrib/tools/python3/src/Include/odictobject.h>
#else
#error "No <odictobject.h> in Python2"
#endif
