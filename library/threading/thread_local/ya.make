LIBRARY()



PEERDIR(
    library/threading/hot_swap
    library/threading/skip_list
)

GENERATE_ENUM_SERIALIZATION(thread_local.h)

SRCS(thread_local.cpp)

END()
