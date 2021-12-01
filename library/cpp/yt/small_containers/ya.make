LIBRARY()

PEERDIR(
    util
    library/cpp/yt/assert
    library/cpp/yt/malloc
)

CHECK_DEPENDENT_DIRS(
    ALLOW_ONLY ALL
    build
    contrib
    library
    util
)

END()

RECURSE_FOR_TESTS(
    unittests
)
