PROGRAM()



NO_LINT()
NO_CHECK_IMPORTS()

PEERDIR(
    contrib/libs/nose/runner
    contrib/python/numpy
)

TEST_SRCS(
    test_regression.py
    test_random.py
)

END()
