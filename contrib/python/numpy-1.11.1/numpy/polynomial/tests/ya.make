PROGRAM()



NO_LINT()
NO_CHECK_IMPORTS()

PEERDIR(
    contrib/libs/nose/runner
    contrib/python/numpy-1.11.1
)

TEST_SRCS(
    test_classes.py
    test_legendre.py
    test_printing.py
    test_laguerre.py
    test_polynomial.py
    test_chebyshev.py
    test_polyutils.py
    test_hermite_e.py
    test_hermite.py
)

END()
