LIBRARY()



SRCS(
    cross_validation.cpp
    dir_helper.cpp
    eval_feature.cpp
    options_helper.cpp
    GLOBAL train_model.cpp
    GLOBAL model_import_snapshot.cpp
)

PEERDIR(
    catboost/private/libs/algo
    catboost/private/libs/algo_helpers
    catboost/libs/column_description
    catboost/private/libs/options
    catboost/libs/data
    catboost/libs/helpers
    catboost/private/libs/data_util
    catboost/private/libs/distributed
    catboost/libs/eval_result
    catboost/private/libs/labels
    catboost/libs/logging
    catboost/libs/loggers
    catboost/libs/metrics
    catboost/libs/model
    catboost/libs/model/model_export
    catboost/libs/fstr
    catboost/libs/overfitting_detector
    catboost/private/libs/pairs
    catboost/private/libs/target
    library/cpp/grid_creator
    library/cpp/json
    library/object_factory
    library/cpp/threading/local_executor
)

IF (USE_MPI)
    CFLAGS(-DUSE_MPI)
ENDIF()

END()
