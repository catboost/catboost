R_MODULE(
    catboostr
    EXPORTS
    catboostr.exports
)



SRCS(
    catboostr.cpp
)

PEERDIR(
    catboost/libs/algo
    catboost/libs/train_lib
    catboost/libs/data
    catboost/libs/fstr
    catboost/libs/helpers
    catboost/libs/logging
    catboost/libs/model
)

IF (OS_WINDOWS)
    LDFLAGS($CURDIR/R.lib)  # TODO: use EXTRALIBS
ENDIF()

END()
