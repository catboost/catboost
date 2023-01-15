LIBRARY()



PEERDIR(
    contrib/libs/zstd
    library/blockcodecs/core
)

SRCS(
    GLOBAL zstd.cpp
)

END()
