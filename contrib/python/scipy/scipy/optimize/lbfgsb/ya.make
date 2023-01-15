PY23_LIBRARY()



IF (PYTHON2)
    PEERDIR(
        contrib/python/numpy/py2/numpy/f2py/src
    )
ELSE()
    PEERDIR(
        contrib/python/numpy/py3/numpy/f2py/src
    )
ENDIF()

NO_COMPILER_WARNINGS()

SRCS(
    lbfgsb.f
    linpack.f
    timer.f

    _lbfgsbmodule.c
)

PY_REGISTER(scipy.optimize._lbfgsb)

END()
