PROGRAM(python)



LICENSE(PSF-2.0)

VERSION(2.7.16)

PEERDIR(
    contrib/tools/python/libpython
    contrib/tools/python/src/Modules/_sqlite
)

END()

RECURSE_FOR_TESTS(
    tests
)
