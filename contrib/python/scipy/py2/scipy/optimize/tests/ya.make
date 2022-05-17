PY2TEST()



VERSION(1.2.3)

ORIGINAL_SOURCE(mirror://pypi/s/scipy/scipy-1.2.3.tar.gz)

SIZE(MEDIUM)

FORK_TESTS()

PEERDIR(
    contrib/python/scipy/py2
    contrib/python/scipy/py2/scipy/conftest
)

NO_LINT()

NO_CHECK_IMPORTS()

TEST_SRCS(
    __init__.py
    test__basinhopping.py
    test__differential_evolution.py
    test__dual_annealing.py
    test__linprog_clean_inputs.py
    test__numdiff.py
    test__remove_redundancy.py
    test__root.py
    test__spectral.py
    test_cobyla.py
    test_constraint_conversion.py
    test_constraints.py
    test_differentiable_functions.py
    test_hessian_update_strategy.py
    test_hungarian.py
    test_lbfgsb_hessinv.py
    test_least_squares.py
    test_linesearch.py
    test_linprog.py
    test_lsq_common.py
    test_lsq_linear.py
    test_minimize_constrained.py
    test_nnls.py
    test_optimize.py
    test_regression.py
    test_slsqp.py
    test_tnc.py
    test_trustregion.py
    test_trustregion_exact.py
    test_trustregion_krylov.py
    test_zeros.py
)

END()
