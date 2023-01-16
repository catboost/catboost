PY2_LIBRARY()



VERSION(0.13.1)

LICENSE(MIT)

PEERDIR(
    contrib/python/importlib-metadata
)

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
    PREFIX contrib/python/pluggy/py2/
    .dist-info/METADATA
    .dist-info/top_level.txt
)

END()

RECURSE_FOR_TESTS(
    tests
)
