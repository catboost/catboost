PY23_LIBRARY()



PY_SRCS(
    __init__.py
    ts_errors.py
    ts_config.py
)

PEERDIR(
    build/plugins/lib/nots/package_manager
)

END()

RECURSE_FOR_TESTS(
    tests
)
