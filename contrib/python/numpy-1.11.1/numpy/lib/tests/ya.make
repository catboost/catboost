PROGRAM()



NO_LINT()
NO_CHECK_IMPORTS()

PEERDIR(
    contrib/libs/nose/runner
    contrib/python/numpy-1.11.1
)

PY_SRCS(
    test_type_check.py
    test_io.py
    test_regression.py
    test_arraysetops.py
    test_nanfunctions.py
    test_utils.py
    test__version.py
    test_arraypad.py
    test_financial.py
    test__iotools.py
    test_stride_tricks.py
    test_polynomial.py
    test_function_base.py
    test_format.py
    test_shape_base.py
    test_recfunctions.py
    test_twodim_base.py
    test_arrayterator.py
    test_index_tricks.py
    test__datasource.py
    test_ufunclike.py
    test_packbits.py
)

END()
