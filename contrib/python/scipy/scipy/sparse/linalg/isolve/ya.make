PY23_LIBRARY()



PEERDIR(
    contrib/python/scipy/scipy/sparse/linalg/isolve/iterative
)

NO_LINT()

PY_SRCS(
    NAMESPACE scipy.sparse.linalg.isolve

    __init__.py
    lsqr.py
    minres.py
    lgmres.py
    iterative.py
    utils.py
    lsmr.py
)

END()
