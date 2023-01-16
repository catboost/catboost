PROGRAM()



NO_LINT()
NO_CHECK_IMPORTS()

PEERDIR(
    contrib/python/nose/runner

    contrib/python/scipy
)

TEST_SRCS(
    test_base.py
    test_construct.py
    test_csc.py
    test_csr.py
    test_extract.py
    test_matrix_io.py
    test_sparsetools.py
    test_spfuncs.py
    test_sputils.py
)

END()
