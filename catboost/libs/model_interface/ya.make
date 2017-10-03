DLL(catboostmodel 1 0 EXPORTS calcer.exports)



PEERDIR(
    catboost/libs/model
)

SRCS(
    model_calcer_wrapper.cpp
)

IF (OS_WINDOWS)
    CFLAGS(-D_WINDLL)
ENDIF()

END()
