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

INCLUDE(${ARCADIA_ROOT}/catboost/cuda/cuda_lib/default_nvcc_flags.make.inc)

ALLOCATOR(LF)


END()
