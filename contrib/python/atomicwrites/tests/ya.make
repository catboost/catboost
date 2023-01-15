PY23_LIBRARY()



PEERDIR(
    contrib/python/atomicwrites
)

TEST_SRCS(
    test_atomicwrites.py
)

NO_LINT()

END()

RECURSE_FOR_TESTS(
    py2
    py3
)
