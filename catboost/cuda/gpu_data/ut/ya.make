UNITTEST(gpu_data_test)



IF (NOT AUTOCHECK)
SRCS(
    test_binarization.cpp
    test_bin_builder.cpp
)
ENDIF()

SIZE(MEDIUM)

SRCS(
    test_data_provider_load.cpp
)

PEERDIR(
    catboost/cuda/gpu_data
    catboost/cuda/data
    catboost/cuda/ut_helpers
    catboost/libs/quantization
)

INCLUDE(${ARCADIA_ROOT}/catboost/cuda/cuda_lib/default_nvcc_flags.make.inc)

ALLOCATOR(LF)

END()
