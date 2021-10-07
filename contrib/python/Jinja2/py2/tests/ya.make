PY2TEST()



PEERDIR(
    contrib/python/Jinja2
)

PY_SRCS(
    TOP_LEVEL
    res/__init__.py
)

DATA(
    arcadia/contrib/python/Jinja2/py2/tests/res
)

RESOURCE_FILES(
    PREFIX contrib/python/Jinja2/py2/tests/
    res/templates/broken.html
    res/templates/foo/test.html
    res/templates/mojibake.txt
    res/templates/syntaxerror.html
    res/templates/test.html
    res/templates2/foo
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
    test_runtime.py
    test_security.py
    test_tests.py
    test_utils.py
)

NO_LINT()

END()
