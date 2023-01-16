PROGRAM()



NO_LINT()
NO_CHECK_IMPORTS()

PEERDIR(
    contrib/python/nose/runner

    contrib/python/scipy
)

TEST_SRCS(
    demo_lgmres.py
    test_iterative.py
    test_lgmres.py
    test_lsmr.py
    test_lsqr.py
    test_utils.py
)

END()
