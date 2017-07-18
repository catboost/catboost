

LIBRARY()

USE_PYTHON()

NO_WERROR()

PEERDIR(
    contrib/python/numpy-1.11.1
)

ADDINCL(
    contrib/python/numpy-1.11.1/numpy/core/include
    GLOBAL contrib/python/numpy-1.11.1/numpy/f2py/src
)

SRCS(
    fortranobject.c
)

END()
