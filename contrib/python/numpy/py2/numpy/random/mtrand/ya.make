PY2_LIBRARY()

LICENSE(BSD-3-Clause)



ADDINCL(
    contrib/python/numpy/include/numpy/core/include
    contrib/python/numpy/include/numpy/core/include/numpy
    contrib/python/numpy/include/numpy/core/src/common
    contrib/python/numpy/include/numpy/random/mtrand
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
    distributions.c
    initarray.c
    randomkit.c
)

PY_SRCS(
    NAMESPACE numpy.random
    CYTHON_C
    mtrand.pyx
)

END()
