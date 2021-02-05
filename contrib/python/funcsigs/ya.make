PY23_LIBRARY()

LICENSE(Apache-2.0)



VERSION(1.0.2)

NO_LINT()

PY_SRCS(
    TOP_LEVEL
    funcsigs/__init__.py
    funcsigs/version.py
)

RESOURCE_FILES(
    PREFIX contrib/python/funcsigs/
    .dist-info/METADATA
)

END()
