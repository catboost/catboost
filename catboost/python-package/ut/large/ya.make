

PYTEST()

SIZE(LARGE)
TAG(ya:dirty ya:fat)
REQUIREMENTS(cpu:all)

TEST_SRCS(test_whl.py)

DEPENDS(catboost/python-package/ut/large/pkg)

NO_LINT()
NO_CHECK_IMPORTS()

END()
