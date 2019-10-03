LIBRARY()



SRCS(
    feature.cpp
    binarizations_manager.cpp
    permutation.cpp
    data_utils.cpp
    leaf_path.cpp
)

PEERDIR(
    catboost/cuda/cuda_lib
    catboost/private/libs/ctr_description
    catboost/libs/data
    catboost/private/libs/data_types
    catboost/libs/model
    catboost/libs/helpers
    catboost/private/libs/options
    catboost/private/libs/feature_estimator
)

GENERATE_ENUM_SERIALIZATION(feature.h)

END()
