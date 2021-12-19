GTEST()



SRCS(
    zig_zag_ut.cpp
    varint_ut.cpp
)

PEERDIR(
    util
    library/cpp/yt/coding
    library/cpp/testing/gtest
)

END()
