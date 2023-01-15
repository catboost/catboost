PY23_LIBRARY()



VERSION(4.4.2)

LICENSE(BSD2)

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
