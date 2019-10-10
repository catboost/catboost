PY23_LIBRARY()



VERSION(19.2.0)

LICENSE(MIT)

NO_LINT()

PY_SRCS(
    TOP_LEVEL
    attr/__init__.py
    attr/_compat.py
    attr/_config.py
    attr/_funcs.py
    attr/_make.py
    attr/_version.py
    attr/converters.py
    attr/exceptions.py
    attr/filters.py
    attr/validators.py
)

RESOURCE_FILES(
    PREFIX contrib/python/attrs/
    .dist-info/METADATA
    .dist-info/top_level.txt
    attr/__init__.pyi
    attr/_version.pyi
    attr/converters.pyi
    attr/exceptions.pyi
    attr/filters.pyi
    attr/py.typed
    attr/validators.pyi
)

END()
