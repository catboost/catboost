PY23_TEST()



PEERDIR(
    contrib/python/pytz
)

SRCDIR(
    contrib/python/pytz/pytz/tests
)

TEST_SRCS(
    test_docs.py
    test_lazy.py
    test_tzinfo.py
)

NO_LINT()

END()
