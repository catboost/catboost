

SUBSCRIBER(
    swarmer
)

LIBRARY()

SRCS(
    subscription.cpp
    wait_all.cpp
    wait_all_or_exception.cpp
    wait_any.cpp
)

PEERDIR(
    library/cpp/threading/future
)

END()

RECURSE_FOR_TESTS(
    ut
)
