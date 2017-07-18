

LIBRARY()

LICENSE(
    BSD2
)

NO_UTIL()

SRCS(
    codec_avx2.c
    lib.c
)

IF (ARCH_X86_64 OR ARCH_I386)
    IF (OS_LINUX OR OS_DARWIN)
        CFLAGS(-mavx2 -std=c11)
    ELSE()
        IF (MSVC)
            CFLAGS(/D__AVX2__=1)
        ENDIF()
    ENDIF()
ENDIF()

END()
