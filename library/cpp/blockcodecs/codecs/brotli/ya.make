LIBRARY()



PEERDIR(
    contrib/libs/brotli/enc
    contrib/libs/brotli/dec
    library/cpp/blockcodecs/core
)

SRCS(
    GLOBAL brotli.cpp
)

END()
