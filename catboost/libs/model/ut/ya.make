UNITTEST(model_ut)



SIZE(MEDIUM)

SRCS(
    model_export_helpers_ut.cpp
    formula_evaluator_ut.cpp
    json_model_export_ut.cpp
    leaf_weights_ut.cpp
    model_metadata_ut.cpp
    model_serialization_ut.cpp
    model_summ_ut.cpp
    shrink_model_ut.cpp
)

PEERDIR(
    catboost/private/libs/algo
    catboost/libs/data
    catboost/libs/model
    catboost/libs/model/ut/lib
    catboost/libs/train_lib
    catboost/private/libs/text_features/ut/lib
    library/json
)

END()
