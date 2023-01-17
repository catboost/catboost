LIBRARY()



PEERDIR(
    contrib/libs/brotli/enc
    contrib/libs/brotli/dec
)

SRCS(
    brotli.cpp
)

END()

RECURSE_FOR_TESTS(
    ut
)
