PY23_LIBRARY()

LICENSE(BSD-2-Clause)

VERSION(0.1.2)



PY_SRCS(
    TOP_LEVEL
    appnope/__init__.py
    appnope/_dummy.py
    appnope/_nope.py
)

NO_LINT()

RESOURCE_FILES(
    PREFIX contrib/python/appnope/
    .dist-info/METADATA
    .dist-info/top_level.txt
)

END()
