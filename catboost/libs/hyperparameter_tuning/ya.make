LIBRARY()



SRCS(
    hyperparameter_tuning.cpp
)

PEERDIR(
    catboost/libs/algo_helpers
    catboost/libs/data
    catboost/libs/helpers
    catboost/libs/train_lib
    catboost/libs/metrics
    catboost/libs/options
    library/json
)

END()
