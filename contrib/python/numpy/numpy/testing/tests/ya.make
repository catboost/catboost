PROGRAM()



NO_LINT()
NO_CHECK_IMPORTS()

PEERDIR(
    contrib/python/nose/runner
    contrib/python/numpy
)

TEST_SRCS(
    test_decorators.py
    test_doctesting.py
    test_utils.py
)

END()
