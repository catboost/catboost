LIBRARY()



SRCS(
    mappers.cpp
    master.cpp
    worker.cpp
)

PEERDIR(
    catboost/libs/algo
    catboost/libs/algo_helpers
    catboost/libs/data_new
    catboost/libs/helpers
    catboost/libs/index_range
    catboost/libs/metrics
    catboost/libs/options
    library/binsaver
    library/par
)

END()
