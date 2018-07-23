R_MODULE(
    catboostr
)
EXPORTS_SCRIPT(catboostr.exports)



SRCS(
    catboostr.cpp
)

PEERDIR(
    catboost/libs/algo
    catboost/libs/data
    catboost/libs/data_util
    catboost/libs/documents_importance
    catboost/libs/fstr
    catboost/libs/eval_result
    catboost/libs/init
    catboost/libs/logging
    catboost/libs/model
    catboost/libs/train_lib
)

IF (OS_WINDOWS)
    LDFLAGS($CURDIR/R.lib)  # TODO: use EXTRALIBS
ENDIF()

IF (NOT OS_WINDOWS)
    ALLOCATOR(LF)
ELSE()
    ALLOCATOR(J)
ENDIF()


END()
