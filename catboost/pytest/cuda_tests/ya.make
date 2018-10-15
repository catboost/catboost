

PYTEST()

IF (NOT SANITIZER_TYPE)

    TEST_SRCS(
        test_gpu.py
    )

ENDIF()

DEPENDS(
    catboost/tools/limited_precision_dsv_diff
)

SIZE(MEDIUM)
REQUIREMENTS(network:full)

PEERDIR(
    catboost/pytest/lib
    catboost/python-package/lib
    contrib/python/numpy
)

DEPENDS(
    catboost/app
)

END()
