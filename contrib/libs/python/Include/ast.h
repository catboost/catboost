#pragma once

#ifdef USE_PYTHON3
#error "No <ast.h> in Python3"
#else
#include <contrib/tools/python/src/Include/ast.h>
#endif
