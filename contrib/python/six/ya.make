

PY23_LIBRARY()

VERSION(1.12.0)

LICENSE(
    MIT
)

PY_SRCS(
    TOP_LEVEL
    six.py
)

RESOURCE_FILES(
    PREFIX contrib/python/six/
    .dist-info/METADATA
)

NO_LINT()

END()
