LIBRARY()



SRCS(
    data_provider.cpp
    load_data.cpp
    feature.cpp
    binarizations_manager.cpp
    binarized_features_meta_info.cpp
    permutation.cpp
    data_utils.cpp
    grid_creator.cpp
    cat_feature_perfect_hash_helper.cpp
)

PEERDIR(
    catboost/libs/ctr_description
    library/threading/local_executor
    library/grid_creator
    catboost/libs/helpers
    catboost/libs/logging
    catboost/libs/data
    catboost/libs/options
    catboost/cuda/utils
)

GENERATE_ENUM_SERIALIZATION(columns.h)
GENERATE_ENUM_SERIALIZATION(feature.h)



END()
