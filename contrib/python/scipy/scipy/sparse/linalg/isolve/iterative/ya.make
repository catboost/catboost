PY23_LIBRARY()



PEERDIR(
    contrib/libs/cblas
    contrib/libs/clapack

    contrib/python/scipy/scipy/_build_utils/src
    contrib/python/numpy/numpy/f2py/src
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
