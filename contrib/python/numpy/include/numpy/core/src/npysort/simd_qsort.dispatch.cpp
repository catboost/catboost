#ifdef USE_PYTHON3
#include <contrib/python/numpy/py3/numpy/core/src/npysort/simd_qsort.dispatch.cpp>
#else
#error #include <contrib/python/numpy/py2/numpy/core/src/npysort/simd_qsort.dispatch.cpp>
#endif
