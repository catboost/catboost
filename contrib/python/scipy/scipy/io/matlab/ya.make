PY23_LIBRARY()

LICENSE(BSD-3-Clause)



ADDINCLSELF()

NO_COMPILER_WARNINGS()

PEERDIR(
    contrib/python/numpy
)

NO_LINT()

PY_SRCS(
    NAMESPACE scipy.io.matlab

    __init__.py
    byteordercodes.py
    mio4.py
    mio5_params.py
    mio5.py
    miobase.py
    mio.py

    CYTHON_C
    mio5_utils.pyx
    mio_utils.pyx
    streams.pyx
)

END()
