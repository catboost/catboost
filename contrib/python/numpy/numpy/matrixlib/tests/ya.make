PROGRAM()



NO_LINT()
NO_CHECK_IMPORTS()

PEERDIR(
    contrib/libs/nose/runner
    contrib/python/numpy
)

TEST_SRCS(
    test_regression.py
    test_defmatrix.py
    test_numeric.py
    test_multiarray.py
)

END()
