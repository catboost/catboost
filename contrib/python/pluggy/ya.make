PY23_LIBRARY()



VERSION(0.9.0)

LICENSE(MIT)

NO_LINT()

PY_SRCS(
    TOP_LEVEL
    pluggy/__init__.py
    pluggy/_tracing.py
    pluggy/_version.py
    pluggy/callers.py
    pluggy/hooks.py
    pluggy/manager.py
)

RESOURCE_FILES(
    PREFIX contrib/python/pluggy/
    .dist-info/METADATA
    .dist-info/top_level.txt
)

END()

RECURSE_FOR_TESTS(
    tests
)
