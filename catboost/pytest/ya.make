

PY3TEST()

TEST_SRCS(
    conftest.py
    large_dist_test.py
    test.py
    test_modes.py
)

DEPENDS(
    catboost/tools/limited_precision_dsv_diff
)

FORK_SUBTESTS()
FORK_TEST_FILES()

SIZE(MEDIUM)

IF(AUTOCHECK)
    SPLIT_FACTOR(240)
    REQUIREMENTS(cpu:4 network:full)
ELSE()
    REQUIREMENTS(cpu:2 network:full)
ENDIF()

PEERDIR(
    catboost/pytest/lib
    catboost/python-package/lib
    contrib/python/numpy
    contrib/python/pandas
)

DEPENDS(
    catboost/app
)

DATA(
    arcadia/catboost/pytest/data
)

IF (OPENSOURCE AND AUTOCHECK)
    INCLUDE(${ARCADIA_ROOT}/catboost//oss/checks/check_deps.inc)
ENDIF()

END()

IF(HAVE_CUDA)
    RECURSE(
    cuda_tests
)
ENDIF()
