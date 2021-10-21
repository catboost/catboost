PY23_LIBRARY()

LICENSE(MIT)



VERSION(0.2.5)

PEERDIR(
    contrib/python/setuptools
)

IF (PYTHON2)
    PEERDIR(
        contrib/python/backports.functools-lru-cache
    )
ENDIF()

PY_SRCS(
    TOP_LEVEL
    wcwidth/__init__.py
    wcwidth/table_wide.py
    wcwidth/table_zero.py
    wcwidth/unicode_versions.py
    wcwidth/wcwidth.py
)

RESOURCE_FILES(
    PREFIX contrib/python/wcwidth/
    .dist-info/METADATA
    .dist-info/top_level.txt
)

NO_LINT()

END()

RECURSE_FOR_TESTS(
    tests
)
