

PY23_LIBRARY()

LICENSE(
    BSD3
)

NO_COMPILER_WARNINGS()

PEERDIR(
    contrib/python/numpy
)

ADDINCL(
    contrib/python/numpy/numpy/core/include
    GLOBAL contrib/python/numpy/numpy/f2py/src
)

SRCS(
    fortranobject.c
)

END()
