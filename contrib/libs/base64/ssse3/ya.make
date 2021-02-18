

LIBRARY()

LICENSE(BSD-2-Clause)

NO_UTIL()

SRCS(
    codec_ssse3.c
    lib.c
)

IF (ARCH_X86_64 OR ARCH_I386)
    IF (MSVC AND NOT CLANG_CL)
        CONLYFLAGS(/D__SSSE3__=1)
    ELSEIF(CLANG_CL)
        CONLYFLAGS(-mssse3)
    ELSE()
        CONLYFLAGS(-mssse3 -std=c11)
    ENDIF()
ENDIF()

END()
