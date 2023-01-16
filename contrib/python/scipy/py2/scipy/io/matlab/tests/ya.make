PROGRAM()



NO_LINT()
NO_CHECK_IMPORTS()

PEERDIR(
    contrib/python/nose/runner

    contrib/python/scipy
)

TEST_SRCS(
    test_byteordercodes.py
    test_mio5_utils.py
    test_miobase.py
    test_mio_funcs.py
    test_mio.py
    test_mio_utils.py
    test_pathological.py
    test_streams.py
)

END()
