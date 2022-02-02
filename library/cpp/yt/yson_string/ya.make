LIBRARY()

SRCS(
    convert.cpp
    string.cpp
)

PEERDIR(
    library/cpp/yt/assert
    library/cpp/yt/coding
    library/cpp/yt/exception
    library/cpp/yt/string
    library/cpp/yt/memory
    library/cpp/yt/misc
)

END()

RECURSE_FOR_TESTS(
    unittests
)
