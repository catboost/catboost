PY23_LIBRARY()




NO_COMPILER_WARNINGS()

PEERDIR(
    contrib/python/numpy/numpy/f2py/src
)

SRCS(
    _slsqpmodule.c
    slsqp_optmz.f
)

PY_REGISTER(scipy.optimize._slsqp)

END()
