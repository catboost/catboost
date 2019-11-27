

PY23_LIBRARY()

NO_COMPILER_WARNINGS()

PEERDIR(
    contrib/python/numpy

    contrib/python/scipy/scipy/sparse/sparsetools
    contrib/python/scipy/scipy/sparse/csgraph
    contrib/python/scipy/scipy/sparse/linalg

    contrib/python/scipy/scipy/_lib
    contrib/python/scipy/scipy/misc
    contrib/python/scipy/scipy/linalg
)

NO_LINT()

PY_SRCS(
    NAMESPACE scipy.sparse

    CYTHON_C
    _csparsetools.pyx

    __init__.py
    compressed.py
    spfuncs.py
    sputils.py
    lil.py
    csr.py
    sparsetools.py
    dok.py
    extract.py
    data.py
    dia.py
    construct.py
    coo.py
    csc.py
    _matrix_io.py
    base.py
    bsr.py
)


END()
