LIBRARY()



PEERDIR(
    contrib/libs/lz4
    contrib/libs/lz4/generated
    library/blockcodecs/core
)

SRCS(
    GLOBAL lz4.cpp
)

END()
