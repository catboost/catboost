PY23_LIBRARY()

LICENSE(BSD-3-Clause)



PEERDIR(
    contrib/python/scipy/scipy/sparse/linalg/eigen/arpack
    contrib/python/scipy/scipy/sparse/linalg/eigen/lobpcg
)

NO_LINT()

PY_SRCS(
    NAMESPACE scipy.sparse.linalg.eigen

    __init__.py
)

END()
