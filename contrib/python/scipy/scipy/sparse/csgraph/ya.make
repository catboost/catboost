

PY23_LIBRARY()

NO_COMPILER_WARNINGS()

PEERDIR(
    contrib/python/numpy

    contrib/python/scipy/scipy/sparse/sparsetools
)

PY_SRCS(
    NAMESPACE scipy.sparse.csgraph

    __init__.py
    _laplacian.py
    _components.py
    _validation.py

    _min_spanning_tree.pyx
    _reordering.pyx
    _shortest_path.pyx
    _tools.pyx
    _traversal.pyx
)

NO_LINT()

END()
