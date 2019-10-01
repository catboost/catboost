LIBRARY()



SRCS(
    binarize_target.cpp
    classification_target_helper.cpp
    data_providers.cpp
    target_converter.cpp
)

PEERDIR(
    catboost/libs/data
    catboost/libs/data_types
    catboost/libs/helpers
    catboost/libs/labels
    catboost/libs/logging
    catboost/libs/metrics
    catboost/libs/options
    catboost/libs/pairs
)

END()
