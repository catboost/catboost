

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

RESOURCE_FILES(
    PREFIX contrib/python/MarkupSafe/
    .dist-info/LICENSE.txt
    .dist-info/METADATA
    .dist-info/top_level.txt
)

END()

RECURSE_FOR_TESTS(
    tests
)
