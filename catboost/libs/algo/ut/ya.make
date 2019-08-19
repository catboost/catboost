UNITTEST(catboost_ut)



SRCS(
    train_ut.cpp
    pairwise_scoring_ut.cpp
    mvs_gen_weights_ut.cpp
    short_vector_ops_ut.cpp
    monotonic_constraints_ut.cpp
)

PEERDIR(
    catboost/libs/algo
    catboost/libs/algo_helpers
    catboost/libs/helpers
    catboost/libs/train_lib
    library/threading/local_executor
)

END()
