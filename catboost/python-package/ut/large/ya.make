

PY3TEST()

SIZE(LARGE)
TAG(ya:dirty ya:fat ya:force_sandbox ya:nofuse)
REQUIREMENTS(cpu:4)

TEST_SRCS(
    conftest.py
    test_whl.py
    run_python3_tests.py
)

PEERDIR(contrib/python/filelock)

DEPENDS(
    catboost/python-package/ut/large/pkg
    catboost/tools/limited_precision_dsv_diff
    catboost/tools/limited_precision_numpy_diff
    catboost/tools/model_comparator
)
DATA(
    arcadia/catboost/pytest/data
    arcadia/catboost/python-package/ut/medium
)

FORK_SUBTESTS()
SPLIT_FACTOR(40)

NO_CHECK_IMPORTS()

END()
