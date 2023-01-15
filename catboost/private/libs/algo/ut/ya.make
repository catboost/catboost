UNITTEST(catboost_ut)



SRCS(
    apply_ut.cpp
    train_ut.cpp
    pairwise_scoring_ut.cpp
    mvs_gen_weights_ut.cpp
    text_collection_builder_ut.cpp
    monotonic_constraints_ut.cpp
    nonsymmetric_index_calcer_ut.cpp
)

PEERDIR(
    catboost/private/libs/algo
    catboost/private/libs/algo_helpers
    catboost/private/libs/text_features/ut/lib
    catboost/libs/data
    catboost/libs/helpers
    catboost/libs/model/ut/lib
    catboost/libs/train_lib
    library/cpp/threading/local_executor
)

END()
