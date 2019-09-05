PY23_LIBRARY()



PEERDIR(
    contrib/python/scipy/scipy/sparse/linalg/eigen/arpack
    contrib/python/scipy/scipy/sparse/linalg/eigen/lobpcg
)

PY_SRCS(
    NAMESPACE scipy.sparse.linalg.eigen

    __init__.py
)

END()
