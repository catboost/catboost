

LIBRARY()

LICENSE(
    BSD2
)

NO_UTIL()

SRCS(
    codec_neon64.c
    lib.c
)

IF (ARCH_AARCH64 OR ARCH_ARM64)
    IF (OS_LINUX OR OS_DARWIN OR OS_ANDROID)
        CONLYFLAGS(-march=armv8-a -std=c11)
    ENDIF()
ENDIF()

END()
