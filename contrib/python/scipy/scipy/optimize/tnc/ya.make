PY23_LIBRARY()

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
