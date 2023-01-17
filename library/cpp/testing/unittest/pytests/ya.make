PY3TEST()



SIZE(MEDIUM)

TEST_SRCS(
    test_tear_down.py
)

PEERDIR(
    library/python/testing/yatest_common
)

DEPENDS(
    library/cpp/testing/unittest/pytests/test_subject
)

END()

RECURSE(
    test_subject
)
