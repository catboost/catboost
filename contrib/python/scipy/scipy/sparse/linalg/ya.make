PY23_LIBRARY()



PEERDIR(
    contrib/python/scipy/scipy/sparse/linalg/isolve
    contrib/python/scipy/scipy/sparse/linalg/dsolve
    contrib/python/scipy/scipy/sparse/linalg/eigen
)

NO_LINT()

PY_SRCS(
    NAMESPACE scipy.sparse.linalg

    __init__.py
    interface.py
    _onenormest.py
    matfuncs.py
    _norm.py
    _expm_multiply.py
)

END()
