UNITTEST(gpu_data_test)



IF (NOT AUTOCHECK)
SRCS(
    test_binarization.cpp
    test_bin_builder.cpp
)
ENDIF()

SRCS(
    test_data_provider_load.cpp
)

PEERDIR(
    catboost/cuda/gpu_data
    catboost/cuda/data
    catboost/cuda/ut_helpers
)

CUDA_NVCC_FLAGS(
    -std=c++11
    -gencode arch=compute_20,code=sm_20
    -gencode arch=compute_35,code=sm_35
    -gencode arch=compute_52,code=sm_52
    -gencode arch=compute_60,code=sm_60
    -gencode arch=compute_61,code=compute_61
    --ptxas-options=-v
)

ALLOCATOR(LF)

END()
