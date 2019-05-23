PY23_LIBRARY()



PEERDIR(
    contrib/python/Jinja2
)

TEST_SRCS(
    conftest.py
    test_api.py
    test_bytecode_cache.py
    test_core_tags.py
    test_debug.py
    test_ext.py
    test_features.py
    test_filters.py
    test_idtracking.py
    test_imports.py
    test_inheritance.py
    test_lexnparse.py
    test_loader.py
    test_nativetypes.py
    test_regression.py
    test_security.py
    test_tests.py
    test_utils.py
)

IF (PYTHON3)
    TEST_SRCS(
        test_asyncfilters.py
        test_async.py
    )
ENDIF()

NO_LINT()

END()

RECURSE_FOR_TESTS(
    py2
    py3
)
