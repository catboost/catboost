PROGRAM()



NO_LINT()
NO_CHECK_IMPORTS()

DATA(
    arcadia/contrib/python/scipy/scipy/interpolate/tests/data
)

PEERDIR(
    contrib/python/nose/runner

    contrib/python/scipy
)

TEST_SRCS(
    test_fitpack2.py
    test_fitpack.py
    test_gil.py
    test_interpnd.py
    test_interpolate.py
    test_interpolate_wrapper.py
    test_ndgriddata.py
    test_polyint.py
    test_rbf.py
    test_regression.py
)

END()
