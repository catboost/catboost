UNITTEST(catboost_ut)



SRCS(
    pairwise_leaves_calculation_ut.cpp
    multiquantile_derivatives_ut.cpp
)

PEERDIR(
    catboost/private/libs/algo_helpers
)

END()
