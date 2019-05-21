PROGRAM()



NO_LINT()
NO_CHECK_IMPORTS()

PEERDIR(
    contrib/python/nose/runner
    contrib/python/numpy
)

TEST_SRCS(
    test_core.py
    test_deprecations.py
    test_extras.py
    test_mrecords.py
    test_old_ma.py
    test_regression.py
    test_subclassing.py
)

END()
