LIBRARY()



CFLAGS(-DONNX_ML=1 -DONNX_NAMESPACE=onnx)

SRCS(
    coreml_helpers.cpp
    ctr_data.cpp
    ctr_provider.cpp
    ctr_value_table.cpp
    features.cpp
    json_model_helpers.cpp
    model.cpp
    online_ctr.cpp
    onnx_helpers.cpp
    static_ctr_provider.cpp
    formula_evaluator.cpp
    model_build_helper.cpp
)

PEERDIR(
    catboost/libs/cat_feature
    catboost/libs/ctr_description
    catboost/libs/helpers
    catboost/libs/logging
    catboost/libs/model/flatbuffers
    catboost/libs/options
    catboost/libs/model/model_export
    contrib/libs/coreml
    contrib/libs/flatbuffers
    contrib/libs/onnx
    library/binsaver
    library/containers/dense_hash
    library/json
    library/svnversion
)

GENERATE_ENUM_SERIALIZATION(ctr_provider.h)
GENERATE_ENUM_SERIALIZATION(split.h)

END()
