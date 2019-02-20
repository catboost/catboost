

PY23_LIBRARY()

VERSION(0.23)

LICENSE(
    BSD3
)

NO_LINT()

PY_SRCS(
    TOP_LEVEL
    markupsafe/_compat.py
    markupsafe/_constants.py
    markupsafe/__init__.py
    markupsafe/_native.py
)

END()
