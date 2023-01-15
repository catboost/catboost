

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
    IF (MSVC AND NOT CLANG_CL)
        CONLYFLAGS(/D__AVX2__=1)
    ELSE()
        CONLYFLAGS(-mavx2 -std=c11)
    ENDIF()
ENDIF()

END()
