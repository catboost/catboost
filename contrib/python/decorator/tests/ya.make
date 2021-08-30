PY23_TEST()



PEERDIR(
    contrib/python/decorator
)

PY_SRCS(
    TOP_LEVEL
    documentation.py
)

TEST_SRCS(
    #documentation.py
    test.py
)

NO_LINT()

END()
