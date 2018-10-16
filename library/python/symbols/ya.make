PY_LIBRARY()



PEERDIR(
    library/python/ctypes
)

SRCS(
    syms.cpp
)

PY_REGISTER(
    library.python.symbols.syms=syms
)

PY_SRCS(
    __init__.py
)

END()
