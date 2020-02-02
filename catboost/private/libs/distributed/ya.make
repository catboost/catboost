LIBRARY()



SRCS(
    mappers.cpp
    master.cpp
    worker.cpp
)

PEERDIR(
    catboost/private/libs/algo
    catboost/private/libs/algo_helpers
    catboost/libs/data
    catboost/libs/helpers
    catboost/private/libs/index_range
    catboost/libs/metrics
    catboost/private/libs/options
    library/binsaver
    library/json
    library/par
)

END()
