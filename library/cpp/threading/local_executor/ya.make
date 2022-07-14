

LIBRARY()

SRCS(
    local_executor.cpp
    tbb_local_executor.cpp
)

PEERDIR(
    contrib/libs/tbb
    library/cpp/threading/future
    library/cpp/deprecated/atomic
)

END()

RECURSE_FOR_TESTS(
    ut
)
