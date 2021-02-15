#ifdef USE_PYTHON3
#include <contrib/python/numpy/py3/numpy/random/src/pcg64/pcg64.h>
#else
#error #include <contrib/python/numpy/py2/numpy/random/src/pcg64/pcg64.h>
#endif
