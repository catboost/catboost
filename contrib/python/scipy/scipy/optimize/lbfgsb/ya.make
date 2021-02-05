PY23_LIBRARY()

LICENSE(BSD-3-Clause)



PEERDIR(
    contrib/python/numpy/py2/numpy/f2py/src
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
