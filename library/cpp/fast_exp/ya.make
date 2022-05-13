LIBRARY()



SRCS(
    fast_exp.h
    fast_exp.cpp
)

IF (ARCH_X86_64 OR ARCH_I386)
    SRC_C_AVX2(fast_exp_avx2.cpp)
    SRC_C_SSE2(fast_exp_sse2.cpp)
ENDIF()

PEERDIR(
    contrib/libs/fmath
)

END()
