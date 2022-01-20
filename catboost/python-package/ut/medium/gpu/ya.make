PY3TEST()



# NO FORK

PEERDIR(
    contrib/python/pandas
    contrib/python/numpy
    contrib/python/scipy
    library/python/pytest
    catboost/python-package/lib
    catboost/pytest/lib
)

SRCDIR(
    catboost/python-package/ut/medium
)

TEST_SRCS(
    gpu/conftest.py
    test.py
)

SIZE(LARGE)

IF (SANITIZER_TYPE)
    TAG(ya:fat ya:not_autocheck)
ELSE()
    TAG(ya:fat ya:yt ya:noretries)
ENDIF()

YT_SPEC(catboost/pytest/cuda_tests/yt_spec.yson)

DATA(
    arcadia/catboost/pytest/data
    arcadia/catboost/python-package
)

DEPENDS(
    catboost/tools/limited_precision_dsv_diff
    catboost/tools/limited_precision_numpy_diff
    catboost/tools/model_comparator
    catboost/python-package/catboost
    catboost/python-package/ut/medium/python_binary
)

IF (OS_LINUX AND NOT ARCH_AARCH64)
    ALLOCATOR(TCMALLOC_256K)
ELSE()
    ALLOCATOR(J)
ENDIF()

END()
