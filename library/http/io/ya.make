LIBRARY()



PEERDIR(
    library/blockcodecs
    library/streams/brotli
    library/streams/bzip2
    library/streams/lzma
)

SRCS(
    chunk.cpp
    compression.cpp
    headers.cpp
    stream.cpp
)

END()
