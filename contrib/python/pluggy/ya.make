PY23_LIBRARY()



VERSION(0.8.0)

LICENSE(MIT)

NO_LINT()

PY_SRCS(
    TOP_LEVEL
    pluggy/__init__.py
    pluggy/_tracing.py
    pluggy/_version.py
    pluggy/callers.py
    pluggy/hooks.py
    pluggy/manager.py
)

END()
