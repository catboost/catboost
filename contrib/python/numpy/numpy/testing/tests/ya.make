PROGRAM()



NO_LINT()
NO_CHECK_IMPORTS()

PEERDIR(
    contrib/libs/nose/runner
    contrib/python/numpy
)

TEST_SRCS(
    test_decorators.py
    test_utils.py
    test_doctesting.py
)

END()
