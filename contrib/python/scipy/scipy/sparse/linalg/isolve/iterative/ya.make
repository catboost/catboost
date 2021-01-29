PY23_LIBRARY()



PEERDIR(
    contrib/libs/cblas
    contrib/libs/clapack

    contrib/python/scipy/scipy/_build_utils/src
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

NO_COMPILER_WARNINGS()

SRCS(
    _iterativemodule.c

    BiCGREVCOM.f
    BiCGSTABREVCOM.f
    CGREVCOM.f
    CGSREVCOM.f
    getbreak.f
    GMRESREVCOM.f
    QMRREVCOM.f
    STOPTEST2.f
)

PY_REGISTER(scipy.sparse.linalg.isolve._iterative)

END()
