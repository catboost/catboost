PY23_LIBRARY()

LICENSE(BSD-3-Clause)

NO_COMPILER_WARNINGS()



ADDINCLSELF()

PEERDIR(
    contrib/python/numpy
)

NO_LINT()

PY_SRCS(
    NAMESPACE scipy.cluster

    __init__.py
    hierarchy.py
    vq.py

    CYTHON_C
    _hierarchy.pyx
    _vq.pyx
)

END()
