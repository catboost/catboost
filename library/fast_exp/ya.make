LIBRARY()



SRCS(
    fast_exp.h
    fast_exp.cpp
)

PEERDIR(
    contrib/libs/fmath
    library/fast_exp/avx2
    library/fast_exp/sse2
)

END()
