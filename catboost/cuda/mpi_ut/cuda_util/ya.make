PROGRAM()



IF(OS_LINUX)

PEERDIR(
    library/unittest
    catboost/cuda/cuda_lib
    catboost/cuda/cuda_util
)

SRCS(
    catboost/cuda/mpi_ut/main.cpp
    catboost/cuda/cuda_util/ut/test_compression_gpu.cpp
    catboost/cuda/cuda_util/ut/test_fill.cpp
    catboost/cuda/cuda_util/ut/test_reduce.cpp
    catboost/cuda/cuda_util/ut/test_reorder_and_partition.cpp
    catboost/cuda/cuda_util/ut/test_scan.cpp
    catboost/cuda/cuda_util/ut/test_segmented_scan.cpp
    catboost/cuda/cuda_util/ut/test_segmented_sort.cpp
    catboost/cuda/cuda_util/ut/test_sort.cpp
    catboost/cuda/cuda_util/ut/test_transform.cpp
)

ELSE()
SRCS(catboost/cuda/mpi_ut/empty_main.cpp)
ENDIF()

CUDA_NVCC_FLAGS(
    -gencode arch=compute_30,code=compute_30
    -gencode arch=compute_35,code=sm_35
    -gencode arch=compute_50,code=compute_50
    -gencode arch=compute_52,code=sm_52
    -gencode arch=compute_60,code=sm_60
    -gencode arch=compute_61,code=sm_61
    -gencode arch=compute_61,code=compute_61
)

ALLOCATOR(LF)


END()
