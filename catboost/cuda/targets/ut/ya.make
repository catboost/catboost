UNITTEST(targets_tests)



IF (NOT AUTOCHECK)
SRCS(
    test_query_cross_entropy.cpp
    test_multi_logit.cpp
)
ENDIF()


PEERDIR(
    catboost/cuda/targets
    catboost/libs/helpers
)

INCLUDE(${ARCADIA_ROOT}/catboost/cuda/cuda_lib/default_nvcc_flags.make.inc)

ALLOCATOR(LF)

END()
