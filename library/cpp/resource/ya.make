LIBRARY()



PEERDIR(
    library/blockcodecs/core
    library/blockcodecs/codecs/zstd
)

SRCS(
    registry.cpp
    resource.cpp
)

END()
