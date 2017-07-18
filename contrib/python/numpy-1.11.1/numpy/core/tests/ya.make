PROGRAM()



NO_LINT()
NO_CHECK_IMPORTS()

PEERDIR(
    contrib/libs/nose/runner
    contrib/python/pytz
    contrib/python/numpy-1.11.1
)

TEST_SRCS(
    test_indexerrors.py
    test_ufunc.py
    test_numerictypes.py
    test_scalarinherit.py
    test_regression.py
    test_nditer.py
    test_abc.py
    test_indexing.py
    test_errstate.py
    test_dtype.py
    test_item_selection.py
    test_umath.py
    test_getlimits.py
    test_machar.py
    test_function_base.py
    test_defchararray.py
    test_datetime.py
    test_umath_complex.py
    test_half.py
    test_shape_base.py
    test_einsum.py
    test_unicode.py
    test_numeric.py
    test_arrayprint.py
    test_scalarprint.py
    test_deprecations.py
    test_mem_overlap.py
    test_print.py
    test_multiarray.py
    test_api.py
    test_scalarmath.py
    test_memmap.py
    test_extint128.py
    test_longdouble.py
    test_records.py
)

END()
