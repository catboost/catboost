LIBRARY()



SRCS(
    blob.cpp
    ref.cpp
    ref_tracked.cpp
)

PEERDIR(
    util
    library/cpp/yt/assert
    library/cpp/yt/misc
    library/cpp/ytalloc/core
    library/cpp/ytalloc/api
)

CHECK_DEPENDENT_DIRS(
    ALLOW_ONLY ALL
    build
    contrib
    library
    util
    library/cpp/yt/assert
    library/cpp/yt/misc
)

END()

RECURSE_FOR_TESTS(
    unittests
)
