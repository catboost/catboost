PY23_LIBRARY()



PEERDIR(
    library/python/symbols/registry
)

SRCS(
    module.cpp
)

PY_REGISTER(
    library.python.symbols.module.syms=syms
)

PY_SRCS(
    __init__.py
)

END()
