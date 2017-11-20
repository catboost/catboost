DLL(catboostmodel 1 0 EXPORTS calcer.exports)



SRCS(
    model_calcer_wrapper.cpp
)

PEERDIR(
    catboost/libs/model
)

IF (OS_WINDOWS)
    CFLAGS(-D_WINDLL)
ENDIF()

END()
