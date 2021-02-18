PY23_LIBRARY()

LICENSE(BSD-3-Clause)



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
