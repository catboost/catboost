// Override _WIN32_WINNT
#undef _WIN32_WINNT
#define _WIN32_WINNT Py_WINVER

#define Py_NO_ENABLE_SHARED

#if !defined(NDEBUG) && !defined(Py_LIMITED_API) && !defined(DISABLE_PYDEBUG)
#define Py_DEBUG
#define GC_NDEBUG
#endif

#include "../PC/pyconfig.h"
