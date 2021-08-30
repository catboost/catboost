PY23_LIBRARY()

LICENSE(BSD-3-Clause)



NO_COMPILER_WARNINGS()

ADDINCLSELF()

ADDINCL(
    FOR cython contrib/python/scipy
)

PEERDIR(
    contrib/python/numpy

    contrib/python/scipy/scipy/interpolate/fitpack
)

SRCS(
    src/_fitpackmodule.c
    src/_interpolate.cpp
    src/dfitpackmodule.c
    src/dfitpack-f2pywrappers.f
)

PY_REGISTER(scipy.interpolate._fitpack)
PY_REGISTER(scipy.interpolate.dfitpack)
PY_REGISTER(scipy.interpolate._interpolate)

NO_LINT()

PY_SRCS(
    NAMESPACE scipy.interpolate

    __init__.py
    _cubic.py
    fitpack2.py
    fitpack.py
    interpolate.py
    interpolate_wrapper.py
    ndgriddata.py
    polyint.py
    rbf.py

    CYTHON_C
    _ppoly.pyx
    interpnd.pyx
)

END()
