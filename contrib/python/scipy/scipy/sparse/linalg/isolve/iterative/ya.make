PY23_LIBRARY()

LICENSE(BSD-3-Clause)



PEERDIR(
    contrib/libs/cblas
    contrib/libs/clapack

    contrib/python/scipy/scipy/_build_utils/src
    contrib/python/numpy
)

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
