PROGRAM()



NO_LINT()
NO_CHECK_IMPORTS()

PEERDIR(
    contrib/python/nose/runner

    contrib/python/scipy
)

TEST_SRCS(
    test_expm_multiply.py
    test_interface.py
    test_matfuncs.py
    test_norm.py
    test_onenormest.py
)

END()
