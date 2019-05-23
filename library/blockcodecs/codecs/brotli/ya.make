LIBRARY()



PEERDIR(
    contrib/libs/brotli/enc
    contrib/libs/brotli/dec
    library/blockcodecs/core
)

SRCS(
    GLOBAL brotli.cpp
)

END()
