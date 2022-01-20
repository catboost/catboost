LIBRARY()

SRCS(
    enum.cpp
    guid.cpp
    string.cpp
)

PEERDIR(
    util
    library/cpp/yt/assert
    library/cpp/yt/misc
)

CHECK_DEPENDENT_DIRS(
    ALLOW_ONLY ALL
    build
    contrib
    library
    util
    library/cpp/yt/assert
    library/cpp/yt/misc
    library/cpp/yt/small_containers
)

END()

RECURSE_FOR_TESTS(
    unittests
)
