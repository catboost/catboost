PY23_TEST()


SUBSCRIBER(g:util-subscribers)

SRCDIR(util/stream)

PY_SRCS(
    NAMESPACE util.stream
    str_ut.pyx
)

TEST_SRCS(
    test_stream.py
)

END()
