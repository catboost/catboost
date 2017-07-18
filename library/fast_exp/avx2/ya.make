LIBRARY()



SRCS(
    fast_exp_avx2.cpp
)

PEERDIR(
    contrib/libs/fmath
)

IF (ARCH_X86_64 OR ARCH_I386)
    IF (OS_LINUX OR OS_DARWIN)
        CFLAGS(-mavx2)
    ELSE()
        IF (MSVC)
            CFLAGS(/D__AVX2__=1)
        ENDIF()
    ENDIF()
ENDIF()

END()
