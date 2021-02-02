PY23_LIBRARY()




NO_COMPILER_WARNINGS()

PEERDIR(
    contrib/python/numpy/py2/numpy/f2py/src
)

SRCS(
    cobyla2.f
    _cobylamodule.c
    trstlp.f
)

PY_REGISTER(scipy.optimize._cobyla)

END()
