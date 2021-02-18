PY23_LIBRARY()

LICENSE(BSD-3-Clause)



VERSION(4.4.2)

NO_LINT()

PY_SRCS(
    TOP_LEVEL
    decorator.py
)

RESOURCE_FILES(
    PREFIX contrib/python/decorator/
    .dist-info/METADATA
    .dist-info/top_level.txt
)

END()

RECURSE_FOR_TESTS(
    tests
)
