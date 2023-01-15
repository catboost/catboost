

LIBRARY()

LICENSE(
    BSD2
)

NO_UTIL()

SRCS(
    codec_plain.c
    lib.c
)

IF (OS_LINUX OR OS_DARWIN)
    CONLYFLAGS(-std=c11)
ENDIF()

END()
