PROGRAM()



NO_LINT()
NO_CHECK_IMPORTS()

PEERDIR(
    contrib/python/nose/runner

    contrib/python/scipy
)

TEST_SRCS(
    test_basic.py
    test_helper.py
    test_import.py
    test_pseudo_diffs.py
    test_real_transforms.py
)

END()
