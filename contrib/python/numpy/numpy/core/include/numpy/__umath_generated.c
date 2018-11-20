#include <Python.h>

#if PY_MAJOR_VERSION == 3
    #include "__umath_generated.c.python3"
#else
    #include "__umath_generated.c.python2"
#endif
