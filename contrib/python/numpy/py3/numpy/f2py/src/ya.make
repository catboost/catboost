

PY3_LIBRARY()

LICENSE(BSD-3-Clause)

NO_COMPILER_WARNINGS()

PEERDIR(
    contrib/python/numpy/py3
)

ADDINCL(
    contrib/python/numpy/py3/numpy/core/include
    GLOBAL contrib/python/numpy/py3/numpy/f2py/src
)

SRCS(
    fortranobject.c
)

END()
