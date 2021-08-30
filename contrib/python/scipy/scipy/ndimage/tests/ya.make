PROGRAM()



NO_LINT()
NO_CHECK_IMPORTS()

PEERDIR(
    contrib/python/nose/runner

    contrib/python/scipy
)

TEST_SRCS(
    test_c_api.py
    test_datatypes.py
    test_filters.py
    test_io.py
    test_measurements.py
    test_ndimage.py
    test_regression.py
)

END()
