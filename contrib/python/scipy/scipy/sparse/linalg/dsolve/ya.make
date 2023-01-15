PY23_LIBRARY()



NO_COMPILER_WARNINGS()

PEERDIR(
    contrib/python/numpy
    contrib/python/scipy/scipy/sparse/linalg/dsolve/SuperLU/SRC
)

NO_LINT()

PY_SRCS(
    NAMESPACE scipy.sparse.linalg.dsolve

    __init__.py
    linsolve.py
    _add_newdocs.py
)

SRCS(
    _superlumodule.c
    _superluobject.c
    _superlu_utils.c
)

PY_REGISTER(scipy.sparse.linalg.dsolve._superlu)

END()
