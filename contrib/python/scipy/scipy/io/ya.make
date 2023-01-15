PY23_LIBRARY()

LICENSE(BSD-3-Clause)



PEERDIR(
    contrib/python/scipy/scipy/io/arff
    contrib/python/scipy/scipy/io/matlab
    contrib/python/scipy/scipy/io/harwell_boeing
)

NO_LINT()

PY_SRCS(
    NAMESPACE scipy.io

    __init__.py
    _fortran.py
    idl.py
    mmio.py
    netcdf.py
    wavfile.py
)

END()
