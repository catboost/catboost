UNITTEST(cuda_lib_ut)

NO_WERROR()



IF (NOT AUTOCHECK)
SRCS(
    test_memory_pool.cpp
    performance_tests.cpp
    test_cuda_buffer.cpp
    test_cuda_manager.cpp
    test_reduce.cpp
    test_reduce_ring.cpp
    test_all_reduce.cpp
    test_serialization.cpp
    test_batch_reduce.cpp
)
ENDIF()

PEERDIR(
    catboost/cuda/cuda_lib
    catboost/cuda/cuda_util
)

SRCS(test_cache.cpp)

INCLUDE(${ARCADIA_ROOT}/catboost/cuda/cuda_lib/default_nvcc_flags.make.inc)

ALLOCATOR(LF)

END()
