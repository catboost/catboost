PY23_LIBRARY()

LICENSE(MIT)



VERSION(1.4.0)

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
