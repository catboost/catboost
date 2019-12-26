PYTEST()



# NO FORK

PEERDIR(
    contrib/python/pandas
    contrib/python/numpy
    contrib/python/scipy/scipy/integrate
    contrib/python/scipy/scipy/sparse
    contrib/python/scipy/scipy/special
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

YT_SPEC(catboost/pytest/cuda_tests/yt_spec.json)

DATA(
    arcadia/catboost/pytest/data
    arcadia/catboost/python-package
)

DEPENDS(
    catboost/tools/limited_precision_dsv_diff
    catboost/tools/model_comparator
    catboost/python-package/catboost
    catboost/python-package/ut/medium/python_binary
)

IF (ARCH_AARCH64 OR OS_WINDOWS)
    ALLOCATOR(J)
ELSE()
    ALLOCATOR(LF)
ENDIF()

END()
