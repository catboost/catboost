LIBRARY()



PEERDIR(
    contrib/libs/fastlz
    contrib/libs/minilzo
    contrib/libs/quicklz

    library/cpp/streams/lz/snappy
    library/cpp/streams/lz/lz4
)

SRCS(
    lz.cpp
)

END()

RECURSE_FOR_TESTS(
    ut
)
