PY23_LIBRARY()

LICENSE(BSD-3-Clause)



NO_COMPILER_WARNINGS()

PEERDIR(
    contrib/python/numpy
    contrib/python/scipy/scipy/fftpack/src
)

NO_LINT()

PY_SRCS(
    NAMESPACE scipy.fftpack

    __init__.py
    basic.py
    fftpack_version.py
    helper.py
    pseudo_diffs.py
    realtransforms.py
)

SRCS(
    convolvemodule.c
    _fftpackmodule.c
)

PY_REGISTER(scipy.fftpack._fftpack)
PY_REGISTER(scipy.fftpack.convolve)

END()
