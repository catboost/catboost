UNITTEST(cuda_lib_ut)

NO_WERROR()



IF (NOT AUTOCHECK)
SRCS(
    test_memory_pool.cpp
    performance_tests.cpp
    test_cuda_buffer.cpp
    test_child_managers.cpp
    test_reduce.cpp
)
ENDIF()

PEERDIR(
    catboost/cuda/cuda_lib
    catboost/cuda/cuda_util
)

SRCS(test_cache.cpp)


CUDA_NVCC_FLAGS(
    -std=c++11
    -gencode arch=compute_30,code=compute_30  -gencode arch=compute_35,code=sm_35  -gencode arch=compute_50,code=compute_50
    -gencode arch=compute_52,code=sm_52
    --ptxas-options=-v
)

ALLOCATOR(LF)

END()
