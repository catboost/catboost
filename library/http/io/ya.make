LIBRARY()



PEERDIR(
    library/blockcodecs
    library/streams/brotli
    library/streams/bzip2
    library/streams/lzma
)

SRCS(
    stream.cpp
    chunk.cpp
    headers.cpp
)

END()
