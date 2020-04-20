

UNITTEST()

SIZE(MEDIUM)

IF (SANITIZER_TYPE)
    TAG(ya:not_autocheck)
ELSE()
    TAG(ya:noretries ya:yt)
ENDIF()

YT_SPEC(catboost/pytest/cuda_tests/yt_spec.json)

SRCS(
    test_auc.cpp
    #test_dcg.cpp # TODO(akhropov): temporarily disabled because it works too slow for MEDIUM test. MLTOOLS-2679.
    test_multi_logit.cpp
    test_query_cross_entropy.cpp
    test_combination.cpp
)

PEERDIR(
    catboost/cuda/cuda_lib
    catboost/cuda/cuda_util
    catboost/cuda/targets
    catboost/cuda/ut_helpers
    catboost/libs/helpers
    catboost/libs/metrics
    library/cpp/accurate_accumulate
    library/cpp/float16
)

INCLUDE(${ARCADIA_ROOT}/catboost/cuda/cuda_lib/default_nvcc_flags.make.inc)

END()
