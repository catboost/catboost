R_MODULE(
    catboostr
)

IF (NOT OS_WINDOWS)
    EXPORTS_SCRIPT(catboostr.exports)
ENDIF()



SRCS(
    catboostr.cpp
)

PEERDIR(
    catboost/libs/cat_feature
    catboost/libs/data
    catboost/libs/eval_result
    catboost/libs/fstr
    catboost/libs/gpu_config/maybe_have_cuda
    catboost/libs/logging
    catboost/libs/model
    catboost/libs/train_lib
    catboost/private/libs/algo
    catboost/private/libs/data_util
    catboost/private/libs/documents_importance
    catboost/private/libs/init
    catboost/private/libs/options
)

IF (HAVE_CUDA)
    PEERDIR(
        catboost/cuda/train_lib
        catboost/libs/model/cuda
    )
ENDIF()

IF (OS_WINDOWS)
    LDFLAGS($CURDIR/R.lib)  # TODO: use EXTRALIBS
ENDIF()

IF (OS_WINDOWS)
    ALLOCATOR(J)
ELSE()
    ALLOCATOR(MIM)
ENDIF()

END()
