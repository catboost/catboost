UNITTEST(catboost_ut)



SRCS(
    apply_ut.cpp
    train_ut.cpp
    pairwise_scoring_ut.cpp
    mvs_gen_weights_ut.cpp
    monotonic_constraints_ut.cpp
    quantile_ut.cpp
)

PEERDIR(
    catboost/libs/algo
    catboost/libs/algo_helpers
    catboost/libs/data
    catboost/libs/helpers
    catboost/libs/model/ut/lib
    catboost/libs/train_lib
    library/threading/local_executor
)

END()
