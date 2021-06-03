PY23_LIBRARY()

LICENSE(BSD-3-Clause)



VERSION(0.2.0)

PY_SRCS(
    TOP_LEVEL
    backcall/__init__.py
    backcall/_signatures.py
    backcall/backcall.py
)

RESOURCE_FILES(
    PREFIX contrib/python/backcall/
    .dist-info/METADATA
    .dist-info/top_level.txt
)

NO_LINT()

END()

RECURSE_FOR_TESTS(
    tests
)
