UNITTEST(method_tests)



IF (NOT AUTOCHECK)
SRCS(
    test_tree_searcher.cpp
    test_pairwise_tree_searcher.cpp
)
ENDIF()


PEERDIR(
    catboost/cuda/gpu_data
    catboost/cuda/data
    catboost/cuda/methods
    catboost/cuda/ut_helpers
)

CUDA_NVCC_FLAGS(
     -gencode arch=compute_30,code=compute_30
       -gencode arch=compute_35,code=sm_35
       -gencode arch=compute_50,code=compute_50
       -gencode arch=compute_52,code=sm_52
       -gencode arch=compute_60,code=compute_60
       -gencode arch=compute_61,code=compute_61
       -gencode arch=compute_61,code=sm_61
       -gencode arch=compute_70,code=sm_70
       -gencode arch=compute_70,code=compute_70
)

ALLOCATOR(LF)

END()
