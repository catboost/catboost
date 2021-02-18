PY23_LIBRARY()

LICENSE(MIT)



VERSION(0.13.1)

NO_LINT()

IF (PYTHON2)
    PEERDIR(
        contrib/python/importlib-metadata
    )
ENDIF()

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
