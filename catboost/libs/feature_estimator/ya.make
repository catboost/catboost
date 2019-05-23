LIBRARY()




SRCS(
    feature_estimator.cpp
)

PEERDIR(
    catboost/libs/helpers
    catboost/libs/model
    catboost/libs/options
    library/threading/local_executor
)


END()
