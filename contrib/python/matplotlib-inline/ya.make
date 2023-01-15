PY3_LIBRARY()



LICENSE(BSD-3-Clause)

VERSION(0.1.2)

PEERDIR(
    contrib/python/traitlets
)

NO_LINT()

NO_CHECK_IMPORTS(
    matplotlib_inline.backend_inline
)

PY_SRCS(
    TOP_LEVEL
    matplotlib_inline/__init__.py
    matplotlib_inline/backend_inline.py
    matplotlib_inline/config.py
)

RESOURCE_FILES(
    PREFIX contrib/python/matplotlib-inline/
    .dist-info/METADATA
    .dist-info/top_level.txt
)

END()
