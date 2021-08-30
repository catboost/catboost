LIBRARY()



SRCS(
    cache.cpp
    thread_safe_cache.cpp
)

END()

RECURSE_FOR_TESTS(
    ut
)
