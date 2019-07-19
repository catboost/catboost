LIBRARY()



SRCS(
    hyperparameter_tuning.cpp
)

PEERDIR(
    catboost/libs/algo
    catboost/libs/data_new
    catboost/libs/helpers
    catboost/libs/train_lib
    catboost/libs/metrics
    catboost/libs/options
    library/json
)

END()
