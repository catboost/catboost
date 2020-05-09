LIBRARY()



SRCS(
    ctr_data.cpp
    ctr_helpers.cpp
    ctr_provider.cpp
    ctr_value_table.cpp
    eval_processing.cpp
    evaluation_interface.cpp
    features.cpp
    GLOBAL model_import_interface.cpp
    model.cpp
    online_ctr.cpp
    scale_and_bias.cpp
    static_ctr_provider.cpp
    model_build_helper.cpp
    cpu/evaluator_impl.cpp
    GLOBAL cpu/formula_evaluator.cpp
    cpu/quantization.cpp
)

PEERDIR(
    catboost/libs/cat_feature
    catboost/private/libs/ctr_description
    catboost/private/libs/text_features
    catboost/libs/helpers
    catboost/libs/logging
    catboost/libs/model/flatbuffers
    catboost/private/libs/options
    contrib/libs/flatbuffers
    library/cpp/binsaver
    library/cpp/containers/dense_hash
    library/cpp/dbg_output
    library/fast_exp
    library/cpp/json
    library/object_factory
    library/svnversion
)

GENERATE_ENUM_SERIALIZATION(ctr_provider.h)
GENERATE_ENUM_SERIALIZATION(enums.h)
GENERATE_ENUM_SERIALIZATION(features.h)
GENERATE_ENUM_SERIALIZATION(split.h)

END()
