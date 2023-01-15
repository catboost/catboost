PY23_LIBRARY()

LICENSE(BSD-3-Clause)



NO_COMPILER_WARNINGS()

PEERDIR(
    contrib/python/numpy
)

SRCS(
    minpack2module.c

    dcsrch.f
    dcstep.f
)

PY_REGISTER(scipy.optimize.minpack2)

END()
