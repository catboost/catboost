LIBRARY()



SRCS(
    apply.cpp
    calc_fstr.cpp
    error_functions.cpp
    eval_helpers.cpp
    features_layout.cpp
    fold.cpp
    full_features.cpp
    greedy_tensor_search.cpp
    index_calcer.cpp
    index_hash_calcer.cpp
    interrupt.cpp
    learn_context.cpp
    online_ctr.cpp
    online_predictor.cpp
    params.cpp
    restorable_rng.cpp
    score_calcer.cpp
    target_classifier.cpp
    train.cpp
    train_model.cpp
    tree_print.cpp
    metric.cpp
    cross_validation.cpp
    helpers.cpp
    cv_data_partition.cpp
)

PEERDIR(
    library/json
    catboost/libs/fstr
    catboost/libs/metrics
    catboost/libs/model
    catboost/libs/helpers
    catboost/libs/logging
    catboost/libs/data
    catboost/libs/overfitting_detector
    library/digest/md5
    library/fast_exp
    library/grid_creator
    library/containers/dense_hash
    library/threading/local_executor
    library/json
)

GENERATE_ENUM_SERIALIZATION(params.h)

GENERATE_ENUM_SERIALIZATION(metric.h)

END()
