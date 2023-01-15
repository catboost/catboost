PY23_LIBRARY()

LICENSE(MIT)



VERSION(21.2.0)

NO_LINT()

PY_SRCS(
    TOP_LEVEL
    attr/__init__.py
    attr/__init__.pyi
    attr/_cmp.py
    attr/_cmp.pyi
    attr/_compat.py
    attr/_config.py
    attr/_funcs.py
    attr/_make.py
    attr/_version_info.py
    attr/_version_info.pyi
    attr/converters.py
    attr/converters.pyi
    attr/exceptions.py
    attr/exceptions.pyi
    attr/filters.py
    attr/filters.pyi
    attr/setters.py
    attr/setters.pyi
    attr/validators.py
    attr/validators.pyi
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
    attr/py.typed
)

END()
