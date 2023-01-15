PY23_LIBRARY()

LICENSE(
    BSD3
)



NO_COMPILER_WARNINGS()

ADDINCL(
    contrib/python/numpy/numpy/core/include
    contrib/python/numpy/numpy/core/include/numpy
    contrib/python/numpy/numpy/core/src/common
)

CFLAGS(
    -DHAVE_CBLAS
    -DHAVE_NPY_CONFIG_H=1
    -DNO_ATLAS_INFO=1
    -D_FILE_OFFSET_BITS=64
    -D_LARGEFILE64_SOURCE=1
    -D_LARGEFILE_SOURCE=1
)

SRCS(
    # lapack_lite/python_xerbla.c is defined in blas.
    lapack_litemodule.c
    umath_linalg.c
)

NO_LINT()

PY_SRCS(
    NAMESPACE numpy.linalg
    __init__.py
    info.py
    linalg.py
)

PY_REGISTER(numpy.linalg._umath_linalg)
PY_REGISTER(numpy.linalg.lapack_lite)


END()
