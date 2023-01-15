PROGRAM()



NO_LINT()
NO_CHECK_IMPORTS()

PEERDIR(
    contrib/python/nose/runner
    contrib/python/numpy
)

TEST_SRCS(
    test_defmatrix.py
    test_interaction.py
    test_masked_matrix.py
    test_matrix_linalg.py
    test_multiarray.py
    test_numeric.py
    test_regression.py
)

END()
