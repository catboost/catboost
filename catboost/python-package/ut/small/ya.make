

PY23_TEST()

SIZE(SMALL)

PEERDIR(
    contrib/python/pandas
    contrib/python/numpy
    contrib/python/scipy/scipy/integrate
    contrib/python/scipy/scipy/sparse
    contrib/python/scipy/scipy/special
    contrib/python/six
    library/python/pytest
    catboost/python-package/lib
    catboost/pytest/lib
)

TEST_SRCS(
    test_pyx_funcs.py
)

DATA(
    arcadia/catboost/pytest/data
    arcadia/catboost/python-package
)

DEPENDS(
    catboost/tools/limited_precision_dsv_diff
    catboost/tools/model_comparator
    catboost/python-package/catboost/no_cuda
    catboost/python-package/ut/medium/python_binary
)

IF (CATBOOST_OPENSOURCE AND AUTOCHECK)
    INCLUDE(${ARCADIA_ROOT}/catboost//oss/checks/check_deps.inc)
ENDIF()

END()
