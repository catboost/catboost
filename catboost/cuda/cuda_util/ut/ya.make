

IF (AUTOCHECK)
    LIBRARY()
ELSE()
    UNITTEST()
ENDIF()

SRCS(
    test_compression_cpu.cpp
    test_compression_gpu.cpp
    test_fill.cpp
    test_reduce.cpp
    test_reorder_and_partition.cpp
    test_scan.cpp
    test_segmented_scan.cpp
    test_segmented_sort.cpp
    test_sort.cpp
    test_transform.cpp
)

SRCS()

PEERDIR(
    catboost/cuda/cuda_util
)

INCLUDE(${ARCADIA_ROOT}/catboost/cuda/cuda_lib/default_nvcc_flags.make.inc)

END()
