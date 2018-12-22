#pragma once

#ifdef USE_PYTHON3
#include <contrib/tools/python3/src/Include/typeslots.h>
#else
#error "No <typeslots.h> in Python2"
#endif
