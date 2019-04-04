

PYTEST()

SIZE(LARGE)
TAG(ya:dirty ya:fat ya:force_sandbox ya:nofuse)
REQUIREMENTS(cpu:all)

TEST_SRCS(
    test_whl.py
    run_python3_tests.py
)

DEPENDS(
    catboost/python-package/ut/large/pkg
    catboost/tools/model_comparator
)
DATA(
    arcadia/catboost/pytest/data
    arcadia/catboost/python-package/ut/medium
)

FORK_SUBTESTS()

NO_LINT()
NO_CHECK_IMPORTS()

END()
