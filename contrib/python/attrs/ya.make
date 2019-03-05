PY23_LIBRARY()



VERSION(18.2.0)

LICENSE(MIT)

NO_LINT()

PY_SRCS(
    TOP_LEVEL
    attr/__init__.py
    attr/_compat.py
    attr/_config.py
    attr/_funcs.py
    attr/_make.py
    attr/converters.py
    attr/exceptions.py
    attr/filters.py
    attr/validators.py
)

END()
