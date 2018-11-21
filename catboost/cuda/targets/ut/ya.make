

UNITTEST(targets_tests)

IF (NOT AUTOCHECK)
SRCS(
    test_auc.cpp
    test_dcg.cpp
    test_multi_logit.cpp
    test_query_cross_entropy.cpp
)
ENDIF()

PEERDIR(
    catboost/cuda/cuda_lib
    catboost/cuda/cuda_util
    catboost/cuda/targets
    catboost/cuda/ut_helpers
    catboost/libs/metrics
    catboost/libs/helpers
)

INCLUDE(${ARCADIA_ROOT}/catboost/cuda/cuda_lib/default_nvcc_flags.make.inc)

END()
