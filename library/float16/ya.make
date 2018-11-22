LIBRARY()



PEERDIR(
    library/float16/float16_avx_impl
)

SRCS(
    float16.cpp
)

END()

NEED_CHECK()
