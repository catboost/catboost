PY23_LIBRARY()

LICENSE(BSD-3-Clause)



NO_COMPILER_WARNINGS()

PEERDIR(
    contrib/python/numpy

    contrib/python/scipy/scipy/odr/odrpack
)

NO_LINT()

PY_SRCS(
    NAMESPACE scipy.odr

    add_newdocs.py
    __init__.py
    models.py
    odrpack.py
)

SRCS(
    __odrpack.c
)

PY_REGISTER(scipy.odr.__odrpack)

END()
