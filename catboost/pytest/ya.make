

PYTEST()

TEST_SRCS(test.py)

FORK_TESTS()
FORK_SUBTESTS()

PEERDIR(
    catboost/pytest/lib
)

DEPENDS(
    catboost/app
)

END()

RECURSE(
    lib
)
