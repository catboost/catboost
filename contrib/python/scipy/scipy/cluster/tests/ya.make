PROGRAM()



NO_LINT()
NO_CHECK_IMPORTS()

PEERDIR(
    contrib/python/nose/runner

    contrib/python/scipy
)

TEST_SRCS(
    hierarchy_test_data.py
    test_hierarchy.py
    test_vq.py
)

END()
