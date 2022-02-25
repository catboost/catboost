LIBRARY()



SRCS(
    guid.cpp
    source_location.cpp
)

PEERDIR(
    contrib/libs/farmhash
    library/cpp/yt/exception
)

CHECK_DEPENDENT_DIRS(
    ALLOW_ONLY ALL
    build
    contrib
    library
    util
    yt/yt/library/small_containers
)

END()

RECURSE_FOR_TESTS(
    unittests
)
