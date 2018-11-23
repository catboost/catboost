LIBRARY()



SRCS(
    cross_validation.cpp
    preprocess.cpp
    GLOBAL train_model.cpp
)

PEERDIR(
    catboost/libs/algo
    catboost/libs/data
    catboost/libs/distributed
    catboost/libs/eval_result
    catboost/libs/fstr
    catboost/libs/helpers
    catboost/libs/labels
    catboost/libs/loggers
    catboost/libs/logging
    catboost/libs/metrics
    catboost/libs/model
    catboost/libs/options
    catboost/libs/pairs
    library/json
    library/object_factory
)


END()
