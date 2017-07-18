PROGRAM()



NO_LINT()
NO_CHECK_IMPORTS()

PEERDIR(
    contrib/libs/nose/runner
    contrib/python/numpy-1.11.1
)

PY_SRCS(
    test_build.py
    test_regression.py
    test_linalg.py
    test_deprecations.py
)

END()
