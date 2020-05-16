

PROGRAM()

IF(OS_LINUX)

PEERDIR(
    library/cpp/unittest
    catboost/cuda/gpu_data
    catboost/cuda/data
    catboost/cuda/methods
    catboost/cuda/ut_helpers
)

SRCS(
    catboost/cuda/mpi_ut/main.cpp
    catboost/cuda/methods/ut/test_tree_searcher.cpp
)

ELSE()
SRCS(catboost/cuda/mpi_ut/empty_main.cpp)
ENDIF()

INCLUDE(${ARCADIA_ROOT}/catboost/cuda/cuda_lib/default_nvcc_flags.make.inc)

END()
