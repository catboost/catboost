

PY3TEST()

SIZE(MEDIUM)

IF(AUTOCHECK)
    REQUIREMENTS(cpu:4 network:full)
ELSE()
    REQUIREMENTS(cpu:2 network:full)
ENDIF()

FORK_SUBTESTS()
SPLIT_FACTOR(40)

PEERDIR(
    contrib/python/tornado/tornado-6
    contrib/python/graphviz
    contrib/python/pandas
    contrib/python/numpy
    contrib/python/scikit-learn
    contrib/python/scipy
    contrib/python/six
    library/python/pytest
    catboost/python-package/lib
    catboost/pytest/lib
)

TEST_SRCS(
    conftest.py
    test.py
    test_whl.py
)

DATA(
    arcadia/catboost/pytest/data
    arcadia/catboost/python-package
    arcadia/library/python/hnsw
)

DEPENDS(
    catboost/tools/limited_precision_dsv_diff
    catboost/tools/limited_precision_numpy_diff
    catboost/tools/model_comparator
    catboost/python-package/catboost/no_cuda
    catboost/python-package/ut/medium/python_binary
    library/python/hnsw/hnsw
)

IF (OPENSOURCE AND AUTOCHECK)
    INCLUDE(${ARCADIA_ROOT}/catboost//oss/checks/check_deps.inc)
ENDIF()

END()
