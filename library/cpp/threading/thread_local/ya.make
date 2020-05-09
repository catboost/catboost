LIBRARY()



PEERDIR(
    library/cpp/threading/hot_swap
    library/cpp/threading/skip_list
)

GENERATE_ENUM_SERIALIZATION(thread_local.h)

SRCS(thread_local.cpp)

END()
