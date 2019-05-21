

PYTEST()

TEST_SRCS(
    test.py
)

FORK_SUBTESTS()

SIZE(SMALL)

DEPENDS(
    catboost/tools/limited_precision_json_diff
)

END()
