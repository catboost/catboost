

IF (AUTOCHECK)
    LIBRARY()
ELSE()
    UNITTEST()
ENDIF()

NO_WERROR()

SRCS(
    performance_tests.cpp
    test_all_reduce.cpp
    test_batch_reduce.cpp
    test_cache.cpp
    test_cuda_buffer.cpp
    test_cuda_manager.cpp
    test_memory_pool.cpp
    test_reduce.cpp
    test_reduce_ring.cpp
    test_serialization.cpp
)

PEERDIR(
    catboost/cuda/cuda_lib
    catboost/cuda/cuda_util
    catboost/libs/helpers
)

INCLUDE(${ARCADIA_ROOT}/catboost/cuda/cuda_lib/default_nvcc_flags.make.inc)

END()
