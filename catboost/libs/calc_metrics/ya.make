LIBRARY()




SRCS(
    calc_metrics.cpp
)

PEERDIR(
    library/cpp/threading/local_executor
    catboost/libs/data
    catboost/libs/metrics
    catboost/private/libs/target
    catboost/private/libs/options
)

END()
