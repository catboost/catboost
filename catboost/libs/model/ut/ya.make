UNITTEST(model_ut)



SRCS(
    formula_evaluator_ut.cpp
    leaf_weights_ut.cpp
    model_metadata_ut.cpp
    model_serialization_ut.cpp
    model_summ_ut.cpp
    model_test_helpers.cpp
    shrink_model_ut.cpp
)

PEERDIR(
    catboost/libs/algo
    catboost/libs/data
    catboost/libs/model
    catboost/libs/train_lib
)

END()
