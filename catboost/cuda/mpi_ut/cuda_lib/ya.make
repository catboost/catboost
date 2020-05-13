

PROGRAM()

IF(OS_LINUX)
SRCS(
    catboost/cuda/mpi_ut/main.cpp
    catboost/cuda/cuda_lib/ut/test_memory_pool.cpp
    catboost/cuda/cuda_lib/ut/performance_tests.cpp
    catboost/cuda/cuda_lib/ut/test_cuda_buffer.cpp
    catboost/cuda/cuda_lib/ut/test_cuda_manager.cpp
    catboost/cuda/cuda_lib/ut/test_reduce.cpp
    catboost/cuda/cuda_lib/ut/test_serialization.cpp
)

PEERDIR(
    library/cpp/unittest
    catboost/cuda/cuda_lib
    catboost/cuda/cuda_util
)
ELSE()
SRCS(catboost/cuda/mpi_ut/empty_main.cpp)
ENDIF()

INCLUDE(${ARCADIA_ROOT}/catboost/cuda/cuda_lib/default_nvcc_flags.make.inc)

END()
