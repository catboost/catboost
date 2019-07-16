DLL(catboostmodel 1 0 )
EXPORTS_SCRIPT(calcer.exports)



SRCS(
    c_api.cpp
)

PEERDIR(
    catboost/libs/cat_feature
    catboost/libs/model
)

IF (OS_WINDOWS)
    CFLAGS(-D_WINDLL)
ENDIF()

END()
