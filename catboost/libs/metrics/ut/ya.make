

UNITTEST()

SIZE(MEDIUM)

PEERDIR(
    catboost/libs/metrics
    catboost/libs/helpers
)

SRCS(
    auc_ut.cpp
    auc_mu_ut.cpp
    brier_score_ut.cpp
    balanced_accuracy_ut.cpp
    dcg_ut.cpp
    fair_loss_ut.cpp
    hamming_loss_ut.cpp
    hinge_loss_ut.cpp
    huber_loss_ut.cpp
    kappa_ut.cpp
    llp_ut.cpp
    median_absolute_error_ut.cpp
    msle_ut.cpp
    normalized_gini_ut.cpp
    precision_recall_at_k_ut.cpp
    quantile_ut.cpp
    smape_ut.cpp
    stochastic_filter_ut.cpp
    zero_one_loss_ut.cpp
    total_f1_ut.cpp
    tweedie_ut.cpp
    pr_auc_ut.cpp
)

END()
