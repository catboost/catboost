PY23_LIBRARY()



PY_SRCS(
    __init__.py
    semver.py
)

END()

RECURSE_FOR_TESTS(
    tests
)
