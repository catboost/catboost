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
    catboost/libs/ctr_description
    catboost/libs/data
    catboost/libs/data_types
    catboost/libs/model
    catboost/libs/helpers
    catboost/libs/options
    catboost/libs/feature_estimator
)

GENERATE_ENUM_SERIALIZATION(feature.h)

END()
