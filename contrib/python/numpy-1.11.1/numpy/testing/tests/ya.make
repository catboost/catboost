PROGRAM()



NO_LINT()
NO_CHECK_IMPORTS()

PEERDIR(
    contrib/libs/nose/runner
    contrib/python/numpy-1.11.1
)

TEST_SRCS(
    test_decorators.py
    test_utils.py
    test_doctesting.py
)

END()
