LIBRARY()



SRCS(
    cat_feature_perfect_hash.cpp
    cat_feature_perfect_hash_helper.cpp
    columns.cpp
    external_columns.cpp
    feature.cpp
    features_layout.cpp
    meta_info.cpp
    objects.cpp
    objects_grouping.cpp
    quantizations_manager.cpp
    target.cpp
    unaligned_mem.cpp
    util.cpp
    weights.cpp
)

PEERDIR(
    library/threading/local_executor

    catboost/libs/ctr_description
    catboost/libs/column_description
    catboost/libs/data_types
    catboost/libs/helpers
    catboost/libs/model
    catboost/libs/options
    catboost/libs/quantization
)

GENERATE_ENUM_SERIALIZATION(target.h)

END()
