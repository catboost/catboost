PROGRAM()



NO_LINT()
NO_CHECK_IMPORTS()

PEERDIR(
    contrib/libs/nose/runner
    contrib/python/numpy
)

TEST_SRCS(
    test_old_ma.py
    test_regression.py
    test_subclassing.py
    test_core.py
    test_extras.py
    test_mrecords.py
)

END()
