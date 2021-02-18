

LIBRARY()

LICENSE(BSD-2-Clause)

NO_UTIL()

SRCS(
    codec_neon32.c
    lib.c
)

IF (OS_LINUX OR OS_DARWIN OR OS_ANDROID)
    CONLYFLAGS(-std=c11)
ENDIF()

END()
