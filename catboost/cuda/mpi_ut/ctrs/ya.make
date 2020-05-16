

PROGRAM()

IF(OS_LINUX)

PEERDIR(
    library/cpp/unittest
    catboost/cuda/ctrs
    catboost/cuda/data
)

SRCS(
    catboost/cuda/mpi_ut/main.cpp
    catboost/cuda/ctrs/ut/test_ctrs.cpp
)
ELSE()
SRCS(catboost/cuda/mpi_ut/empty_main.cpp)
ENDIF()

INCLUDE(${ARCADIA_ROOT}/catboost/cuda/cuda_lib/default_nvcc_flags.make.inc)

END()
