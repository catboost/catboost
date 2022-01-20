PROGRAM()



NO_LINT()
NO_CHECK_IMPORTS()

PEERDIR(
    contrib/python/nose/runner

    contrib/python/scipy
)

TEST_SRCS(
    test__basinhopping.py
    test_cobyla.py
    test__differential_evolution.py
    test_hungarian.py
    test_lbfgsb_hessinv.py
    test_least_squares.py
    test_linesearch.py
    test_linprog.py
    test_lsq_common.py
    test_lsq_linear.py
    test_minpack.py
    test_nnls.py
    test_nonlin.py
    test__numdiff.py
    test_optimize.py
    test_regression.py
    test__root.py
    test_slsqp.py
    test__spectral.py
    test_tnc.py
    test_trustregion.py
    test_zeros.py
)

END()
