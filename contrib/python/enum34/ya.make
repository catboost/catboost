

PY_LIBRARY() # Backport from Python 3.

LICENSE(
    BSD
)

VERSION(
    1.1.6
)

PY_SRCS(
    TOP_LEVEL
    enum/__init__.py
    enum/enum.py
)

RESOURCE_FILES(
    PREFIX contrib/python/enum34/
    .dist-info/METADATA
    .dist-info/top_level.txt
)

NO_LINT()

END()

RECURSE(
    tests
)
