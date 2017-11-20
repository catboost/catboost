LIBRARY()



SRCS(
    plot.cpp
    apply.cpp
    error_functions.cpp
    features_layout.cpp
    fold.cpp
    full_features.cpp
    greedy_tensor_search.cpp
    index_calcer.cpp
    index_hash_calcer.cpp
    learn_context.cpp
    logger.cpp
    model_build_helper.cpp
    online_ctr.cpp
    online_predictor.cpp
    restorable_rng.cpp
    score_calcer.cpp
    split.cpp
    target_classifier.cpp
    train.cpp
    tree_print.cpp
    cross_validation.cpp
    helpers.cpp
    cv_data_partition.cpp
    GLOBAL train_model.cpp
)

PEERDIR(
    catboost/libs/data
    catboost/libs/helpers
    catboost/libs/logging
    catboost/libs/metrics
    catboost/libs/model
    catboost/libs/overfitting_detector
    catboost/libs/params
    catboost/libs/tensorboard_logger
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

GENERATE_ENUM_SERIALIZATION(logger.h)

END()
