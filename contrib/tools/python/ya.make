PROGRAM(python)



VERSION(2.7.16)

PEERDIR(
    contrib/tools/python/libpython
    contrib/tools/python/src/Modules/_sqlite
)

END()

RECURSE_FOR_TESTS(
    tests
)
