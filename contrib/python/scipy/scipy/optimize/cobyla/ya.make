PY23_LIBRARY()

LICENSE(BSD-3-Clause)



NO_COMPILER_WARNINGS()

PEERDIR(
    contrib/python/numpy
)

SRCS(
    cobyla2.f
    _cobylamodule.c
    trstlp.f
)

PY_REGISTER(scipy.optimize._cobyla)

END()
