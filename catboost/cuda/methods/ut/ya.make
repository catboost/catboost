UNITTEST(method_tests)



IF (NOT AUTOCHECK)
SRCS(
    test_tree_searcher.cpp
)
ENDIF()


PEERDIR(
    catboost/cuda/gpu_data
    catboost/cuda/data
    catboost/cuda/methods
    catboost/cuda/ut_helpers
)

CUDA_NVCC_FLAGS(
     -gencode arch=compute_30,code=compute_30  -gencode arch=compute_35,code=sm_35  -gencode arch=compute_50,code=compute_50
    -gencode arch=compute_52,code=sm_52
    -gencode arch=compute_60,code=sm_60
    -gencode arch=compute_61,code=compute_61
)

ALLOCATOR(LF)

END()
