

PY23_LIBRARY()

PY_SRCS(
    __init__.py
    CYTHONIZE_PY
    strings.py
)

PEERDIR(
    library/python/func
    contrib/python/six
)

STYLE_PYTHON()

END()

RECURSE(
    ut
)
