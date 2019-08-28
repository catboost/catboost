LIBRARY()



SRCS(
    cross_validation.cpp
    eval_feature.cpp
    options_helper.cpp
    GLOBAL train_model.cpp
    GLOBAL model_import_snapshot.cpp
)

PEERDIR(
    catboost/libs/algo
    catboost/libs/algo_helpers
    catboost/libs/column_description
    catboost/libs/options
    catboost/libs/data_new
    catboost/libs/helpers
    catboost/libs/data_util
    catboost/libs/distributed
    catboost/libs/eval_result
    catboost/libs/labels
    catboost/libs/logging
    catboost/libs/loggers
    catboost/libs/metrics
    catboost/libs/model
    catboost/libs/model/model_export
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
