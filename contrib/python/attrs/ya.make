PY23_LIBRARY()



VERSION(20.3.0)

LICENSE(MIT)

NO_LINT()

PY_SRCS(
    TOP_LEVEL
    attr/__init__.py
    attr/_compat.py
    attr/_config.py
    attr/_funcs.py
    attr/_make.py
    attr/_version_info.py
    attr/converters.py
    attr/exceptions.py
    attr/filters.py
    attr/setters.py
    attr/validators.py
)

IF (PYTHON3)
    PY_SRCS(
        TOP_LEVEL
        attr/_next_gen.py
    )
ENDIF()

RESOURCE_FILES(
    PREFIX contrib/python/attrs/
    .dist-info/METADATA
    .dist-info/top_level.txt
    attr/__init__.pyi
    attr/_version_info.pyi
    attr/converters.pyi
    attr/exceptions.pyi
    attr/filters.pyi
    attr/py.typed
    attr/setters.pyi
    attr/validators.pyi
)

END()
