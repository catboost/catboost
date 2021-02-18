PY23_LIBRARY()

LICENSE(BSD-3-Clause)

NO_COMPILER_WARNINGS()

PEERDIR(
    contrib/python/numpy
)



SRCS(
    moduleTNC.c
    tnc.c
)

PY_REGISTER(scipy.optimize.moduleTNC)

END()
