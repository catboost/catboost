

LIBRARY()

LICENSE(
    BSD-2-Clause
    MIT
)

LICENSE_TEXTS(.yandex_meta/licenses.list.txt)

NO_UTIL()

SRCS(
    codec_avx2.c
    lib.c
)

IF (ARCH_X86_64 OR ARCH_I386)
    IF (MSVC AND NOT CLANG_CL)
        CONLYFLAGS(/D__AVX2__=1)
    ELSEIF (CLANG_CL)
        CONLYFLAGS(-mavx2)
    ELSE()
        CONLYFLAGS(
            -mavx2
            -std=c11
        )
    ENDIF()
ENDIF()

END()
