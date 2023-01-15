LIBRARY()



SRCS(
    mappers.cpp
    master.cpp
    worker.cpp
)

PEERDIR(
    catboost/libs/data
    catboost/libs/helpers
    catboost/libs/metrics
    catboost/private/libs/algo
    catboost/private/libs/algo/approx_calcer
    catboost/private/libs/algo_helpers
    catboost/private/libs/index_range
    catboost/private/libs/options
    library/cpp/binsaver
    library/cpp/json
    library/par
)

END()
