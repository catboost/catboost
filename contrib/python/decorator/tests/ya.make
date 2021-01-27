PY23_LIBRARY()



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

RECURSE_FOR_TESTS(
    py2
    py3
)
