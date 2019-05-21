PROGRAM()



NO_LINT()
NO_CHECK_IMPORTS()

PEERDIR(
    contrib/python/nose/runner
    contrib/python/pytz
    contrib/python/numpy
)

TEST_SRCS(
    _locales.py
    test_abc.py
    test_api.py
    test_arrayprint.py
    test_datetime.py
    test_defchararray.py
    test_deprecations.py
    test_dtype.py
    test_einsum.py
    test_errstate.py
    test_extint128.py
    test_function_base.py
    test_getlimits.py
    test_half.py
    test_indexerrors.py
    test_indexing.py
    test_item_selection.py
    test_longdouble.py
    test_machar.py
    test_mem_overlap.py
    test_memmap.py
    test_multiarray.py
    test_nditer.py
    test_numeric.py
    test_numerictypes.py
    test_overrides.py
    test_print.py
    test_records.py
    test_regression.py
    test_scalar_ctors.py
    test_scalarbuffer.py
    test_scalarinherit.py
    test_scalarmath.py
    test_scalarprint.py
    test_shape_base.py
    test_ufunc.py
    test_umath.py
    test_umath_complex.py
    test_unicode.py
)

END()
