PY_PROGRAM()



NO_LINT()
NO_CHECK_IMPORTS()

PEERDIR(
    contrib/python/nose/runner
    contrib/python/numpy
)

PY_SRCS(
    test__datasource.py
    test__iotools.py
    test__version.py
    test_arraypad.py
    test_arraysetops.py
    test_arrayterator.py
    test_financial.py
    test_format.py
    test_function_base.py
    test_histograms.py
    test_index_tricks.py
    test_io.py
    test_mixins.py
    test_nanfunctions.py
    test_packbits.py
    test_polynomial.py
    test_recfunctions.py
    test_regression.py
    test_shape_base.py
    test_stride_tricks.py
    test_twodim_base.py
    test_type_check.py
    test_ufunclike.py
    test_utils.py
)

END()
