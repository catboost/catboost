

PY23_LIBRARY()

PY_SRCS(__init__.py)

PEERDIR(
    library/python/func
    library/python/strings
    library/python/windows
)

NO_LINT()

END()

RECURSE_FOR_TESTS(test)
