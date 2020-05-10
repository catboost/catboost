LIBRARY()



SRCS(
    parallel_sort.cpp
)

PEERDIR(
    catboost/private/libs/index_range
    library/cpp/threading/local_executor
)

END()
