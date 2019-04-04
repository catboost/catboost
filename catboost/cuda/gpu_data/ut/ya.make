

IF (AUTOCHECK)
    LIBRARY()
ELSE()
    UNITTEST()
    SIZE(MEDIUM)
    ALLOCATOR(LF)
ENDIF()

SRCS(
    test_bin_builder.cpp
    test_binarization.cpp
)

PEERDIR(
    catboost/cuda/gpu_data
    catboost/cuda/data
    catboost/cuda/ut_helpers
    catboost/libs/helpers
    catboost/libs/quantization
)

INCLUDE(${ARCADIA_ROOT}/catboost/cuda/cuda_lib/default_nvcc_flags.make.inc)

END()
