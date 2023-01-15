

LIBRARY()

LICENSE(
    BSD2
)

NO_UTIL()

SRCS(
    codec_ssse3.c
    lib.c
)

IF (ARCH_X86_64 OR ARCH_I386)
    IF (OS_LINUX OR OS_DARWIN OR CLANG_CL)
        CONLYFLAGS(-mssse3 -std=c11)
    ELSE()
        IF (MSVC)
            CONLYFLAGS(/D__SSSE3__=1)
        ENDIF()
    ENDIF()
ENDIF()

END()
