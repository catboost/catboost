DLL(catboost)
EXPORTS_SCRIPT(catboost.exports)



SRCS(
    catboost_api.cpp

)

PEERDIR(
    catboost/libs/algo
    catboost/libs/column_description
    catboost/libs/data_new
    catboost/libs/data_util
    catboost/libs/helpers
    catboost/libs/init
    catboost/libs/labels
    catboost/libs/logging
    catboost/libs/metrics
    catboost/libs/model
    catboost/libs/options
    catboost/libs/target
    catboost/libs/train_lib
    library/grid_creator
    library/json
    library/logger
)

IF(HAVE_CUDA)
    PEERDIR(
        catboost/cuda/train_lib
    )
ENDIF()

ALLOCATOR(LF)

INCLUDE(${ARCADIA_ROOT}/catboost/cuda/cuda_lib/default_nvcc_flags.make.inc)

IF (OS_WINDOWS)
    CFLAGS(-D_WINDLL)
ENDIF()

END()
