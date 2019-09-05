PROGRAM()

NO_COMPILER_WARNINGS()



NO_LINT()
NO_CHECK_IMPORTS()

PEERDIR(
    contrib/python/nose/runner

    contrib/python/scipy
)

TEST_SRCS(
    test_banded_ode_solvers.py
    test_bvp.py
    test_integrate.py
    test_odeint_jac.py
    test_quadpack.py
    test_quadrature.py
)

END()
