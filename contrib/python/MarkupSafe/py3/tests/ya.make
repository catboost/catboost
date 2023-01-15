PY3TEST()



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
