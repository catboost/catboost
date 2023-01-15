PY23_LIBRARY()

LICENSE(BSD-3-Clause)



NO_COMPILER_WARNINGS()

PEERDIR(
    contrib/python/numpy/py2/numpy/f2py/src
)

SRCS(
    _slsqpmodule.c
    slsqp_optmz.f
)

PY_REGISTER(scipy.optimize._slsqp)

END()
