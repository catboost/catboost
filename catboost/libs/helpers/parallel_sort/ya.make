LIBRARY()



SRCS(
    parallel_sort.cpp
)

PEERDIR(
    catboost/private/libs/index_range
    library/threading/local_executor
)

END()
