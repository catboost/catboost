

LIBRARY()

SRCS(
    subscription.cpp
    wait_all.cpp
    wait_all_or_exception.cpp
    wait_any.cpp
)

END()

RECURSE_FOR_TESTS(
    ut
)
