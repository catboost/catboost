LIBRARY()



SRCS(
    cross_validation.cpp
    GLOBAL train_model.cpp
)

PEERDIR(
    catboost/libs/algo
    catboost/libs/options
    catboost/libs/data_new
    catboost/libs/distributed
    catboost/libs/eval_result
    catboost/libs/helpers
    catboost/libs/labels
    catboost/libs/logging
    catboost/libs/loggers
    catboost/libs/metrics
    catboost/libs/model
    catboost/libs/fstr
    catboost/libs/overfitting_detector
    catboost/libs/pairs
    catboost/libs/target
    library/grid_creator
    library/json
    library/object_factory
    library/threading/local_executor
)


END()
