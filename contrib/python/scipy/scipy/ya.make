PY23_LIBRARY()



PEERDIR(
    contrib/python/scipy/scipy/special
    contrib/python/scipy/scipy/integrate
    contrib/python/scipy/scipy/optimize
    contrib/python/scipy/scipy/interpolate
    contrib/python/scipy/scipy/spatial
    contrib/python/scipy/scipy/fftpack
    contrib/python/scipy/scipy/signal
    contrib/python/scipy/scipy/ndimage
    contrib/python/scipy/scipy/stats
    contrib/python/scipy/scipy/io
    contrib/python/scipy/scipy/constants
    contrib/python/scipy/scipy/cluster
    contrib/python/scipy/scipy/odr
)

NO_LINT()

PY_SRCS(
    NAMESPACE scipy

    __init__.py
    __config__.py
    version.py
)

END()
