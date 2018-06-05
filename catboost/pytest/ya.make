

PYTEST()

TEST_SRCS(test.py)

IF (NOT AUTOCHECK AND HAVE_CUDA)
    TEST_SRCS(test_gpu.py)
ENDIF()

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
