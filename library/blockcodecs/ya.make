LIBRARY()



PEERDIR(
    library/blockcodecs/core
    library/blockcodecs/codecs/brotli
    library/blockcodecs/codecs/bzip
    library/blockcodecs/codecs/fastlz
    library/blockcodecs/codecs/legacy_zstd06
    library/blockcodecs/codecs/lz4
    library/blockcodecs/codecs/lzma
    library/blockcodecs/codecs/snappy
    library/blockcodecs/codecs/zlib
    library/blockcodecs/codecs/zstd
)

SRCS(
    codecs.cpp
    stream.cpp
)

END()
