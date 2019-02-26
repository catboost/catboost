UNITTEST(catboost_ut)



SRCS(
    train_ut.cpp
    pairwise_leaves_calculation_ut.cpp
    pairwise_scoring_ut.cpp
    mvs_gen_weights_ut.cpp
)

PEERDIR(
    catboost/libs/algo
    catboost/libs/train_lib
)

END()
