R_MODULE(
    catboostr
)
EXPORTS_SCRIPT(catboostr.exports)



SRCS(
    catboostr.cpp
)

PEERDIR(
    catboost/libs/algo
    catboost/libs/train_lib
    catboost/libs/data
    catboost/libs/fstr
    catboost/libs/documents_importance
    catboost/libs/helpers
    catboost/libs/logging
    catboost/libs/model
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
