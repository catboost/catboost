PY23_LIBRARY()



NO_COMPILER_WARNINGS()

IF (PYTHON2)
    PEERDIR(
        contrib/python/numpy/py2/numpy/f2py/src
    )
ELSE()
    PEERDIR(
        contrib/python/numpy/py3/numpy/f2py/src
    )
ENDIF()

SRCS(
    _slsqpmodule.c
    slsqp_optmz.f
)

PY_REGISTER(scipy.optimize._slsqp)

END()
