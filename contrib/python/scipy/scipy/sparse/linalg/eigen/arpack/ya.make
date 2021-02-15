PY23_LIBRARY()

LICENSE(BSD-3-Clause)



NO_COMPILER_WARNINGS()

PEERDIR(
    contrib/python/scipy/scipy/sparse/linalg/eigen/arpack/ARPACK
    contrib/python/numpy
)

SRCS(
    _arpackmodule.c
    _arpack-f2pywrappers.f
)

NO_LINT()

PY_SRCS(
    NAMESPACE scipy.sparse.linalg.eigen.arpack

    __init__.py
    arpack.py
)

PY_REGISTER(scipy.sparse.linalg.eigen.arpack._arpack)

END()
