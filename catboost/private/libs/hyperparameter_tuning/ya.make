LIBRARY()



SRCS(
    hyperparameter_tuning.cpp
)

PEERDIR(
    catboost/private/libs/algo_helpers
    catboost/libs/data
    catboost/libs/helpers
    catboost/libs/train_lib
    catboost/libs/metrics
    catboost/private/libs/options
    library/json
)

END()
