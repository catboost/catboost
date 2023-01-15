PY23_TEST()



PEERDIR(
    contrib/python/atomicwrites
)

TEST_SRCS(
    test_atomicwrites.py
)

NO_LINT()

END()
