LIBRARY()



PEERDIR(
    contrib/python/scipy-0.18.1/scipy/special
    contrib/python/scipy-0.18.1/scipy/integrate
    contrib/python/scipy-0.18.1/scipy/optimize
    contrib/python/scipy-0.18.1/scipy/interpolate
    contrib/python/scipy-0.18.1/scipy/spatial
    contrib/python/scipy-0.18.1/scipy/fftpack
    contrib/python/scipy-0.18.1/scipy/signal
    contrib/python/scipy-0.18.1/scipy/ndimage
    contrib/python/scipy-0.18.1/scipy/stats
    contrib/python/scipy-0.18.1/scipy/io
    contrib/python/scipy-0.18.1/scipy/constants
    contrib/python/scipy-0.18.1/scipy/cluster
    contrib/python/scipy-0.18.1/scipy/odr
)


PY_SRCS(
    NAMESPACE scipy

    __init__.py
    __config__.py
    version.py
)

END()
