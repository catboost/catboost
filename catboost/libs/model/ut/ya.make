UNITTEST(model_ut)



SRCS(
    formula_evaluator_ut.cpp
    model_serialization_ut.cpp
    leaf_weights_ut.cpp
)

PEERDIR(
    catboost/libs/model
    catboost/libs/algo
    catboost/libs/train_lib
)

END()
