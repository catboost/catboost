PROGRAM()



NO_LINT()
NO_CHECK_IMPORTS()

PEERDIR(
    contrib/python/nose/runner

    contrib/python/scipy
)

TEST_SRCS(
    test_common.py
    test_doccer.py
    test_pilutil.py
)

END()
