PY23_LIBRARY()

LICENSE(MIT)



PEERDIR(
    contrib/python/six
)

TEST_SRCS(
    test_six.py
)

NO_LINT()

END()

RECURSE_FOR_TESTS(
    py2
    py3
)
