PROGRAM()



NO_LINT()
NO_CHECK_IMPORTS()

PEERDIR(
    contrib/libs/nose/runner
    contrib/python/numpy
)

PY_SRCS(
    test_build.py
    test_regression.py
    test_linalg.py
    test_deprecations.py
)

END()
