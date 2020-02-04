LIBRARY()



SRCS(
    binarize_target.cpp
    classification_target_helper.cpp
    data_providers.cpp
    target_converter.cpp
)

PEERDIR(
    catboost/libs/data
    catboost/private/libs/data_types
    catboost/libs/helpers
    catboost/private/libs/labels
    catboost/libs/logging
    catboost/libs/metrics
    catboost/private/libs/options
    catboost/private/libs/pairs
    library/json
)

END()
