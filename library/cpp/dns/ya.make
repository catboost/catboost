LIBRARY()



SRCS(
    cache.cpp
    thread.cpp
    magic.cpp
)

END()

RECURSE_FOR_TESTS(
    ut
)
