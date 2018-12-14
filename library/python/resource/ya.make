PY23_LIBRARY()



PEERDIR(
    contrib/python/six
)

PY_SRCS(__init__.py)

END()

RECURSE_FOR_TESTS(ut)
