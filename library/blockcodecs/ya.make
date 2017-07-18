LIBRARY()



PEERDIR(
    contrib/libs/fastlz
    contrib/libs/libbz2
    contrib/libs/lz4
    contrib/libs/lz4/generated
    contrib/libs/lzmasdk
    contrib/libs/snappy
    contrib/libs/zlib
    contrib/libs/zstd01
    contrib/libs/zstd06
    contrib/libs/zstd
)

SRCS(
    legacy_zstd01.cpp
    legacy_zstd06.cpp
    codecs.cpp
    stream.cpp
)

END()
