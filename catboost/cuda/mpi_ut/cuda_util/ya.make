PROGRAM()



IF(OS_LINUX)

PEERDIR(
    library/cpp/unittest
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

INCLUDE(${ARCADIA_ROOT}/catboost/cuda/cuda_lib/default_nvcc_flags.make.inc)

IF (ARCH_AARCH64)
    ALLOCATOR(J)
ELSE()
    ALLOCATOR(LF)
ENDIF()

END()
