LIBRARY()



PEERDIR(
    library/logger
)

IF (OS_WINDOWS)
    NO_WERROR()
ENDIF()

SRCS(
    common.cpp
    global.cpp
    rty_formater.cpp
)

END()
