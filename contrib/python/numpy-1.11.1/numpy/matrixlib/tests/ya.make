PROGRAM()



NO_LINT()
NO_CHECK_IMPORTS()

PEERDIR(
    contrib/libs/nose/runner
    contrib/python/numpy-1.11.1
)

TEST_SRCS(
    test_regression.py
    test_defmatrix.py
    test_numeric.py
    test_multiarray.py
)

END()
