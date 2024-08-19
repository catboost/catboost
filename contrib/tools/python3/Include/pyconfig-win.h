// Override _WIN32_WINNT
#undef _WIN32_WINNT
#define _WIN32_WINNT Py_WINVER

#define Py_NO_ENABLE_SHARED

#include "../PC/pyconfig.h"
