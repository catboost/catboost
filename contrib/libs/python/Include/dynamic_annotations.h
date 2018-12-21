#pragma once

#ifdef USE_PYTHON3
#include <contrib/tools/python3/src/Include/dynamic_annotations.h>
#else
#error "No <dynamic_annotations.h> in Python2"
#endif
