

LIBRARY()

SRCS(
    auc.cpp
    auc_mu.cpp
    balanced_accuracy.cpp
    brier_score.cpp
    classification_utils.cpp
    dcg.cpp
    description_utils.cpp
    hinge_loss.cpp
    kappa.cpp
    llp.cpp
    metric.cpp
    optimal_const_for_loss.cpp
    pfound.cpp
    precision_recall_at_k.cpp
    sample.cpp
    caching_metric.cpp
)

PEERDIR(
    catboost/private/libs/data_types
    catboost/libs/eval_result
    catboost/libs/helpers
    catboost/libs/logging
    catboost/private/libs/options
    catboost/libs/helpers/parallel_sort
    library/cpp/binsaver
    library/cpp/containers/2d_array
    library/cpp/containers/stack_vector
    library/cpp/dot_product
    library/cpp/threading/local_executor
)

GENERATE_ENUM_SERIALIZATION(enums.h)

END()
