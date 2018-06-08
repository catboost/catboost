

PYTEST()

TEST_SRCS(
    test.py
    test_gpu.py
)

FORK_TESTS()
FORK_SUBTESTS()

SIZE(MEDIUM)

PEERDIR(
    catboost/pytest/lib
    catboost/python-package/lib
    contrib/python/numpy
)

DEPENDS(
    catboost/app
)

END()

RECURSE(
    lib
)
