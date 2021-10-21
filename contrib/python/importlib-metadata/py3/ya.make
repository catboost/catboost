PY3_LIBRARY()



VERSION(3.10.1)

LICENSE(Apache-2.0)

PEERDIR(
    library/python/resource
)

NO_LINT()

PY_SRCS(
    TOP_LEVEL
    importlib_metadata/__init__.py
    importlib_metadata/_collections.py
    importlib_metadata/_compat.py
    importlib_metadata/_functools.py
    importlib_metadata/_itertools.py
)

RESOURCE_FILES(
    PREFIX contrib/python/importlib-metadata/py3/
    .dist-info/METADATA
    .dist-info/top_level.txt
    importlib_metadata/py.typed
)

END()
