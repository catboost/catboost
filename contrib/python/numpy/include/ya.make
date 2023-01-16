

LIBRARY()

LICENSE(BSD-3-Clause)

ADDINCL(
    GLOBAL contrib/python/numpy/include/numpy/core/include
    GLOBAL contrib/python/numpy/include/numpy/core/include/numpy
    GLOBAL contrib/python/numpy/include/numpy/core/src/common
    GLOBAL contrib/python/numpy/include/numpy/core/src/npymath
    GLOBAL FOR cython contrib/python/numpy/include/numpy/core/include
    GLOBAL FOR cython contrib/python/numpy/include/numpy/core/include/numpy

)

END()
