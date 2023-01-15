PY23_LIBRARY()



VERSION(1.3.0)

LICENSE(MIT)

NO_LINT()

PY_SRCS(
    TOP_LEVEL
    atomicwrites/__init__.py
)

RESOURCE_FILES(
    PREFIX contrib/python/atomicwrites/
    .dist-info/METADATA
    .dist-info/top_level.txt
)

END()

RECURSE_FOR_TESTS(
    tests
)
