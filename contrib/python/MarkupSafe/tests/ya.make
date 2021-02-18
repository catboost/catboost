PY23_LIBRARY()

LICENSE(BSD-3-Clause)



PEERDIR(
    contrib/python/MarkupSafe
)

TEST_SRCS(
    conftest.py
    test_escape.py
    test_exception_custom_html.py
    test_leak.py
    test_markupsafe.py
)

NO_LINT()

END()

RECURSE_FOR_TESTS(
    py2
    py3
)
