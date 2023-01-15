PY23_LIBRARY()



VERSION(1.0.2)

LICENSE(Apache)

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
