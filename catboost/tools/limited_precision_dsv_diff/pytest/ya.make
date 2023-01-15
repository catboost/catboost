

PYTEST()

TEST_SRCS(
    test.py
)

FORK_SUBTESTS()

SIZE(SMALL)

DEPENDS(
    catboost/tools/limited_precision_dsv_diff
)

END()
