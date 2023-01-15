

PY3_LIBRARY()

LICENSE(BSD-3-Clause)

VERSION(2.0.1)

PY_SRCS(
    TOP_LEVEL
    markupsafe/__init__.py
    markupsafe/_native.py
    markupsafe/_speedups.pyi
)

SRCS(
    markupsafe/_speedups.c
)

PY_REGISTER(
    markupsafe._speedups
)

NO_LINT()

NO_COMPILER_WARNINGS()

RESOURCE_FILES(
    PREFIX contrib/python/MarkupSafe/py3/
    .dist-info/METADATA
    .dist-info/top_level.txt
    markupsafe/py.typed
)

END()

RECURSE_FOR_TESTS(
    tests
)
