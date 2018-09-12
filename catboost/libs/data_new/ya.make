LIBRARY()



SRCS(
    cat_feature_perfect_hash.cpp
    cat_feature_perfect_hash_helper.cpp
    columns.cpp
    external_columns.cpp
    feature.cpp
    features_layout.cpp
    meta_info.cpp
    quantizations_manager.cpp
)

PEERDIR(
    library/threading/local_executor

    catboost/libs/column_description
    catboost/libs/data_types
    catboost/libs/ctr_description
    catboost/libs/column_description
    catboost/libs/data_types
    catboost/libs/helpers
    catboost/libs/model
    catboost/libs/options
    catboost/libs/quantization
)

END()
