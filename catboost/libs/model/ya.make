LIBRARY()



SRCS(
    coreml_helpers.cpp
    ctr_provider.cpp
    model.cpp
    model_calcer.cpp
    online_ctr.cpp
    static_ctr_provider.cpp
    tensor_struct.cpp
)

PEERDIR(
    catboost/libs/ctr_description
    catboost/libs/helpers
    catboost/libs/cat_feature
    catboost/libs/logging
    library/json
    contrib/libs/coreml
)

GENERATE_ENUM_SERIALIZATION(split.h)

END()
