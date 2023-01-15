PY23_LIBRARY()

LICENSE(BSD-3-Clause)



NO_COMPILER_WARNINGS()

PEERDIR(
    contrib/python/numpy
)

SRCS(
    _ctest.c
    nd_image.c
    ni_filters.c
    ni_fourier.c
    ni_interpolation.c
    ni_measure.c
    ni_morphology.c
    ni_support.c
)

PY_SRCS(
    NAMESPACE scipy.ndimage

    _cytest.pyx
    _ni_label.pyx
)

PY_REGISTER(scipy.ndimage._nd_image)

NO_LINT()

END()
