

PY23_LIBRARY()

VERSION(1.1.1)

LICENSE(BSD3)

PY_SRCS(
    TOP_LEVEL
    markupsafe/_compat.py
    markupsafe/_constants.py
    markupsafe/__init__.py
    markupsafe/_native.py
)

NO_LINT()

END()

RECURSE_FOR_TESTS(
    tests
)
