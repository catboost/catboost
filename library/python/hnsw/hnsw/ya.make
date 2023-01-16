PY_ANY_MODULE(_hnsw PREFIX "")



IF (PYTHON_CONFIG MATCHES "python3" OR USE_SYSTEM_PYTHON MATCHES "3.")
    PYTHON3_MODULE()
    EXPORTS_SCRIPT(hnsw3.exports)
ELSE()
    PYTHON2_MODULE()
    EXPORTS_SCRIPT(hnsw.exports)
ENDIF()


PYTHON2_ADDINCL()

PEERDIR(
    contrib/python/numpy/include # add only headers for dynamic linking
    library/cpp/hnsw/helpers
    library/cpp/hnsw/index
    library/cpp/hnsw/index_builder
    library/cpp/hnsw/logging
    library/cpp/online_hnsw/base
    library/cpp/online_hnsw/dense_vectors
)

SRCS(
    helpers.cpp
)

# have to disable them because cython's numpy integration uses deprecated numpy API
NO_COMPILER_WARNINGS()

BUILDWITH_CYTHON_CPP(
    _hnsw.pyx
    --module-name _hnsw
)

END()
