R_MODULE(
    catboostr
)
EXPORTS_SCRIPT(catboostr.exports)



SRCS(
    catboostr.cpp
)

PEERDIR(
    catboost/libs/algo
    catboost/libs/cat_feature
    catboost/libs/data_util
    catboost/libs/data_new
    catboost/libs/documents_importance
    catboost/libs/fstr
    catboost/libs/gpu_config/maybe_have_cuda
    catboost/libs/eval_result
    catboost/libs/init
    catboost/libs/logging
    catboost/libs/model
    catboost/libs/options
    catboost/libs/train_lib
    catboost/libs/options
)

IF (HAVE_CUDA)
    PEERDIR(
        catboost/cuda/train_lib
    )
ENDIF()

IF (OS_WINDOWS)
    LDFLAGS($CURDIR/R.lib)  # TODO: use EXTRALIBS
ENDIF()

IF (NOT OS_WINDOWS)
    ALLOCATOR(LF)
ELSE()
    ALLOCATOR(J)
ENDIF()


END()
