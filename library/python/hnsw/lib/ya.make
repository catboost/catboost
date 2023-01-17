PY23_LIBRARY()



SRCDIR(library/python/hnsw/hnsw)

PEERDIR(
    library/cpp/hnsw/helpers
    library/cpp/hnsw/index
    library/cpp/hnsw/index_builder
    library/cpp/hnsw/logging
    library/cpp/online_hnsw/base
    library/cpp/online_hnsw/dense_vectors
    contrib/python/numpy
    contrib/python/six
)

IF (PYTHON2)
    PEERDIR(
        contrib/deprecated/python/enum34
    )
ENDIF()

SRCS(
    library/python/hnsw/hnsw/helpers.cpp
)

# have to disable them because cython's numpy integration uses deprecated numpy API
NO_COMPILER_WARNINGS()

PY_SRCS(
    NAMESPACE hnsw
    __init__.py
    _hnsw.pyx
    hnsw.py
)

END()
