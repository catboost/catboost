UNITTEST(model_ut)



SRCS(
    model_export_helpers_ut.cpp
    formula_evaluator_ut.cpp
    json_model_export_ut.cpp
    leaf_weights_ut.cpp
    model_metadata_ut.cpp
    model_serialization_ut.cpp
    model_summ_ut.cpp
    model_test_helpers.cpp
    shrink_model_ut.cpp
)

PEERDIR(
    catboost/libs/algo
    catboost/libs/data_new
    catboost/libs/data_new/ut/lib
    catboost/libs/model
    catboost/libs/train_lib
)

END()
