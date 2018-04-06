UNITTEST(cuda_util_ut)

NO_WERROR()



IF (NOT AUTOCHECK)
SRCS(
    test_fill.cpp
    test_transform.cpp
    test_scan.cpp
    test_sort.cpp
    test_reduce.cpp
    test_segmented_scan.cpp
    test_reorder_and_partition.cpp
    test_compression_gpu.cpp
    test_segmented_sort.cpp
)
ENDIF()

SRCS(test_compression_cpu.cpp)

PEERDIR(
    catboost/cuda/cuda_util
)


CUDA_NVCC_FLAGS(
     -gencode arch=compute_30,code=compute_30  -gencode arch=compute_35,code=sm_35  -gencode arch=compute_50,code=compute_50
    -gencode arch=compute_52,code=sm_52
    -gencode arch=compute_60,code=sm_60
    -gencode arch=compute_61,code=compute_61
)

ALLOCATOR(LF)


END()
