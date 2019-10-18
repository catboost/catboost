R_MODULE(
    catboostr
)
EXPORTS_SCRIPT(catboostr.exports)



SRCS(
    catboostr.cpp
)

PEERDIR(
    catboost/private/libs/algo
    catboost/libs/cat_feature
    catboost/private/libs/data_util
    catboost/libs/data
    catboost/private/libs/documents_importance
    catboost/libs/fstr
    catboost/libs/gpu_config/maybe_have_cuda
    catboost/libs/eval_result
    catboost/private/libs/init
    catboost/libs/logging
    catboost/libs/model
    catboost/private/libs/options
    catboost/libs/train_lib
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

IF (ARCH_AARCH64 OR OS_WINDOWS)
    ALLOCATOR(J)
ELSE()
    ALLOCATOR(LF)
ENDIF()

END()
