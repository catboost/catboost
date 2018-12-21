PY23_LIBRARY()



NO_LINT()

PY_SRCS(
    TOP_LEVEL
    funcsigs/__init__.py
    funcsigs/version.py
)

RESOURCE(
    .dist-info/METADATA /fs/contrib/python/funcsigs/.dist-info/METADATA
)

END()
