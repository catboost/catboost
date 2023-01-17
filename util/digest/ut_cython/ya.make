PY23_TEST()


SUBSCRIBER(g:util-subscribers)

SRCDIR(util/digest)

PY_SRCS(
    NAMESPACE util.digest
    multi_ut.pyx
)

TEST_SRCS(
    test_digest.py
)

END()
