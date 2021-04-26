

PY2TEST()

TEST_SRCS(
    conftest.py
    test_gpu.py
)

IF(SANITIZER_TYPE)
    TAG(ya:not_autocheck)
ENDIF()

DEPENDS(
    catboost/tools/limited_precision_dsv_diff
)

SIZE(MEDIUM)

TAG(ya:yt)

YT_SPEC(catboost/pytest/cuda_tests/yt_spec.yson)

IF(AUTOCHECK)
    FORK_SUBTESTS()
ENDIF()

PEERDIR(
    catboost/pytest/lib
    catboost/python-package/lib
    contrib/python/numpy
)

DEPENDS(
    catboost/app
)

DATA(
    arcadia/catboost/pytest/data
)

END()
