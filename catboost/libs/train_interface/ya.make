DLL(catboost)
EXPORTS_SCRIPT(catboost.exports)



SRCS(
    catboost_api.cpp

)

PEERDIR(
    catboost/private/libs/algo
    catboost/libs/column_description
    catboost/libs/data
    catboost/private/libs/data_util
    catboost/libs/helpers
    catboost/private/libs/init
    catboost/private/libs/labels
    catboost/libs/logging
    catboost/libs/metrics
    catboost/libs/model
    catboost/private/libs/options
    catboost/private/libs/target
    catboost/libs/train_lib
    library/cpp/grid_creator
    library/cpp/threading/local_executor
    library/cpp/json
    library/cpp/logger
)

IF(HAVE_CUDA)
    PEERDIR(
        catboost/cuda/train_lib
        catboost/libs/model/cuda
    )
ENDIF()

IF (ARCH_AARCH64 OR OS_WINDOWS)
    ALLOCATOR(J)
ELSE()
    ALLOCATOR(LF)
ENDIF()

INCLUDE(${ARCADIA_ROOT}/catboost/cuda/cuda_lib/default_nvcc_flags.make.inc)

IF (OS_WINDOWS)
    CFLAGS(-D_WINDLL)
ENDIF()

END()
