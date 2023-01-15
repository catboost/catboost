PY23_LIBRARY()



PEERDIR(
    contrib/python/pytz
)

TEST_SRCS(
    test_docs.py
    test_lazy.py
    test_tzinfo.py
)

NO_LINT()

END()

RECURSE_FOR_TESTS(
    py2
    py3
)
