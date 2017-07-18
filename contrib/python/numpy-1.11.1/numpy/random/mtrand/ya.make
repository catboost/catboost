LIBRARY()



ADDINCL(
    contrib/python/numpy-1.11.1/numpy/core/src/private
    contrib/python/numpy-1.11.1/numpy/core/include
    contrib/python/numpy-1.11.1/numpy/core/include/numpy
    contrib/python/numpy-1.11.1/numpy/random/mtrand
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
