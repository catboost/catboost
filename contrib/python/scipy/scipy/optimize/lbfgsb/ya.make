PY23_LIBRARY()




PEERDIR(
    contrib/python/numpy/numpy/f2py/src
)

NO_COMPILER_WARNINGS()

SRCS(
    lbfgsb.f
    linpack.f
    timer.f

    _lbfgsbmodule.c
)

PY_REGISTER(scipy.optimize._lbfgsb)

END()
