LIBRARY()



SRCS(
    fast_exp.h
    fast_exp.cpp
)

IF (ARCH_X86_64)
    SRC_CPP_AVX2(fast_exp_avx2.cpp)
    SRC_CPP_SSE2(fast_exp_sse2.cpp)
ENDIF()

PEERDIR(
    contrib/libs/fmath
)

END()
