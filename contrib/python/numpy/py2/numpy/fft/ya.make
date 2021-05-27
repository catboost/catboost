PY2_LIBRARY()

LICENSE(BSD-3-Clause)



ADDINCL(
    contrib/python/numpy/py2/numpy/core/include
    contrib/python/numpy/py2/numpy/core/include/numpy
    contrib/python/numpy/py2/numpy/core/src/common
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
    fftpack.c
    fftpack_litemodule.c
)

NO_LINT()

PY_SRCS(
    NAMESPACE numpy.fft
    __init__.py
    fftpack.py
    helper.py
    info.py
)

PY_REGISTER(numpy.fft.fftpack_lite)

END()
