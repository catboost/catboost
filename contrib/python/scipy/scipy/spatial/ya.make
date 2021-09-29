PY23_LIBRARY()

LICENSE(BSD-3-Clause)



NO_COMPILER_WARNINGS()

ADDINCL(
    contrib/libs/qhull

    contrib/python/scipy/scipy/spatial
    contrib/python/scipy/scipy/spatial/ckdtree/src
)

PEERDIR(
    contrib/libs/qhull
    contrib/python/scipy/scipy/spatial/ckdtree/src
)

NO_LINT()

PY_SRCS(
    NAMESPACE scipy.spatial

    __init__.py
    distance.py
    kdtree.py
    _plotutils.py
    _procrustes.py
    _spherical_voronoi.py

    CYTHON_C
    qhull.pyx

    CYTHON_CPP
    ckdtree.pyx
)

SRCS(
    src/distance_wrap.c
)

PY_REGISTER(scipy.spatial._distance_wrap)

END()
