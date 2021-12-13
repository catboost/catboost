PROGRAM()



NO_LINT()
NO_CHECK_IMPORTS()

PEERDIR(
    contrib/python/nose/runner

    contrib/python/scipy
)

TEST_SRCS(
    test__gcutils.py
    test__threadsafety.py
    test_tmpdirs.py
    test__util.py
    test__version.py
)

END()
