PY23_LIBRARY()



NO_COMPILER_WARNINGS()

PEERDIR(
    contrib/python/scipy/scipy/sparse/linalg/eigen/arpack/ARPACK
)

IF (PYTHON2)
    PEERDIR(
        contrib/python/numpy/py2/numpy/f2py/src
    )
ELSE()
    PEERDIR(
        contrib/python/numpy/py3/numpy/f2py/src
    )
ENDIF()

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
