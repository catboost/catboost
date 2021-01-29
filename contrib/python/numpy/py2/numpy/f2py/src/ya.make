

PY23_LIBRARY()

LICENSE(BSD-3-Clause)

NO_COMPILER_WARNINGS()

PEERDIR(
    contrib/python/numpy
)

ADDINCL(
    contrib/python/numpy/py2/numpy/core/include
    GLOBAL contrib/python/numpy/py2/numpy/f2py/src
)

SRCS(
    fortranobject.c
)

END()
