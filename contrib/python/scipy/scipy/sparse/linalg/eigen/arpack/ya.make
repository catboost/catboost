PY23_LIBRARY()



NO_COMPILER_WARNINGS()


PEERDIR(
    contrib/python/numpy/numpy/f2py/src
    contrib/python/scipy/scipy/sparse/linalg/eigen/arpack/ARPACK
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
