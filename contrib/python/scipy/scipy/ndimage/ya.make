PY23_LIBRARY()

LICENSE(BSD-3-Clause)



NO_COMPILER_WARNINGS()

PEERDIR(
    contrib/python/scipy/scipy/ndimage/src
)

NO_LINT()

PY_SRCS(
    NAMESPACE scipy.ndimage

    __init__.py
    filters.py
    fourier.py
    interpolation.py
    io.py
    measurements.py
    morphology.py
    _ni_support.py
)

END()
