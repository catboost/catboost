PY23_LIBRARY()

LICENSE(Apache-2.0)



VERSION(2.1.1)

NO_LINT()

PEERDIR(
    library/python/resource
)

IF (PYTHON2)
    PEERDIR(
        contrib/python/pathlib2
        contrib/python/contextlib2
        contrib/python/configparser
    )
ENDIF()

PY_SRCS(
    TOP_LEVEL
    importlib_metadata/__init__.py
    importlib_metadata/_compat.py
)

RESOURCE_FILES(
    PREFIX contrib/python/importlib-metadata/
    .dist-info/METADATA
    .dist-info/top_level.txt
)

END()
