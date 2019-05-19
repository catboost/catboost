PY_LIBRARY()



NO_WSHADOW()

PEERDIR(
    contrib/tools/python/lib
    library/resource
)

CFLAGS(-DCYTHON_REGISTER_ABCS=0)

NO_PYTHON_INCLUDES()

PY_SRCS(
    entry_points.py
    TOP_LEVEL
    __res.pyx
    sitecustomize.pyx
)

END()
