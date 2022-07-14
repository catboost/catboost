PY23_LIBRARY()

LICENSE(Service-Py23-Proxy)



ADDINCL(
    GLOBAL contrib/python/numpy/include/numpy/core/include
    GLOBAL contrib/python/numpy/include/numpy/core/include/numpy
    GLOBAL contrib/python/numpy/include/numpy/f2py/src
    GLOBAL FOR cython contrib/python/numpy/include/numpy/core/include
    GLOBAL FOR cython contrib/python/numpy/include/numpy/core/include/numpy
)

IF (PYTHON2)
    PEERDIR(contrib/python/numpy/py2)
ELSE()
    PEERDIR(contrib/python/numpy/py3)
ENDIF()

NO_LINT()

END()

RECURSE(
    py2
    py3
)
