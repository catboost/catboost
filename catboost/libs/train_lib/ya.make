LIBRARY()



SRCS(
    cross_validation.cpp
    preprocess.cpp
    GLOBAL train_model.cpp
)

PEERDIR(
    catboost/libs/data
    catboost/libs/algo
    catboost/libs/distributed
    catboost/libs/helpers
    catboost/libs/logging
    catboost/libs/loggers
    catboost/libs/metrics
    catboost/libs/model
    catboost/libs/fstr
    catboost/libs/overfitting_detector
    library/binsaver
    library/containers/2d_array
    library/containers/dense_hash
    library/digest/md5
    library/dot_product
    library/fast_exp
    library/fast_log
    library/grid_creator
    library/json
    library/object_factory
    library/threading/local_executor
)


END()
