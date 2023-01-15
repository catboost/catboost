GTEST()



SRCS(
    stack_allocator_ut.cpp
    stack_guards_ut.cpp
    stack_pool_ut.cpp
    stack_ut.cpp
    stack_utils_ut.cpp
)

PEERDIR(
    library/cpp/coroutine/engine
)

END()