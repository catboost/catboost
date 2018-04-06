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
)

CUDA_NVCC_FLAGS(
    -gencode arch=compute_30,code=compute_30
    -gencode arch=compute_35,code=sm_35
    -gencode arch=compute_50,code=compute_50
    -gencode arch=compute_52,code=sm_52
    -gencode arch=compute_60,code=sm_60
    -gencode arch=compute_61,code=compute_61
)

ALLOCATOR(LF)

END()
