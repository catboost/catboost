

PY23_LIBRARY()

LICENSE(MIT)

VERSION(1.16.0)

PY_SRCS(
    TOP_LEVEL
    six.py
)

RESOURCE_FILES(
    PREFIX contrib/python/six/
    .dist-info/METADATA
    .dist-info/top_level.txt
)

NO_LINT()

END()

RECURSE_FOR_TESTS(
    tests
)
