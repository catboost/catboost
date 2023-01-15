PY_PROGRAM()



NO_LINT()
NO_CHECK_IMPORTS()

PEERDIR(
    contrib/python/nose/runner
    contrib/python/numpy
    contrib/python/pytest
)

PY_SRCS(
    test_build.py
    test_deprecations.py
    test_linalg.py
    test_regression.py
)

END()
