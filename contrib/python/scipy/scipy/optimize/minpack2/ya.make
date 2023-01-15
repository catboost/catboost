PY23_LIBRARY()



NO_COMPILER_WARNINGS()

PEERDIR(
    contrib/python/numpy/py2/numpy/f2py/src
)

SRCS(
    minpack2module.c

    dcsrch.f
    dcstep.f
)

PY_REGISTER(scipy.optimize.minpack2)

END()
