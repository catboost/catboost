PY23_LIBRARY()



NO_COMPILER_WARNINGS()

PEERDIR(
    contrib/python/scipy/scipy/fftpack/src
)

IF (PYTHON2)
    PEERDIR(
        contrib/python/numpy/py2/numpy/f2py/src
    )
ELSE()
    PEERDIR(
        contrib/python/numpy/py3/numpy/f2py/src
    )
ENDIF()

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
