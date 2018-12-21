PROGRAM(python)



NO_WSHADOW()

PEERDIR(
    contrib/tools/python/libpython
)

END()

RECURSE_FOR_TESTS(
    tests
)
