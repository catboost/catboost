#pragma once

#ifdef USE_PYTHON3
#include <contrib/tools/python3/Include/internal/pycore_critical_section.h>
#else
#error "No <internal/pycore_critical_section.h> in Python2"
#endif
