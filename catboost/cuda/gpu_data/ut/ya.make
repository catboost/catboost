

IF (AUTOCHECK)
    LIBRARY()
ELSE()
    UNITTEST()
    SIZE(MEDIUM)
    IF (ARCH_AARCH64 OR OS_WINDOWS)
        ALLOCATOR(J)
    ELSE()
        ALLOCATOR(LF)
    ENDIF()
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
    catboost/private/libs/quantization
)

INCLUDE(${ARCADIA_ROOT}/catboost/cuda/cuda_lib/default_nvcc_flags.make.inc)

END()
