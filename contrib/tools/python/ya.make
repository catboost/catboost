PROGRAM(python)



PEERDIR(
    contrib/tools/python/libpython
    contrib/tools/python/src/Modules/_sqlite
)

END()

RECURSE_FOR_TESTS(
    tests
)
