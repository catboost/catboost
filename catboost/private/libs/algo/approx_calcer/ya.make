LIBRARY()



SRCS(
    approx_calcer_multi.cpp
    eval_additive_metric_with_leaves.cpp
    leafwise_approx_calcer.cpp
)

PEERDIR(
    catboost/libs/helpers
    catboost/private/libs/algo_helpers
    catboost/private/libs/options
)

END()

