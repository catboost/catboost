LIBRARY()



SRCS(
    async_row_processor.cpp
    cat_feature_perfect_hash.cpp
    cat_feature_perfect_hash_helper.cpp
    columns.cpp
    data_provider.cpp
    external_columns.cpp
    feature_index.cpp
    features_layout.cpp
    meta_info.cpp
    objects.cpp
    objects_grouping.cpp
    quantized_features_info.cpp
    target.cpp
    unaligned_mem.cpp
    util.cpp
    weights.cpp
)

PEERDIR(
    library/dbg_output
    library/threading/future
    library/threading/local_executor

    catboost/libs/column_description
    catboost/libs/data_types
    catboost/libs/helpers
    catboost/libs/model
    catboost/libs/options
    catboost/libs/quantization
)

GENERATE_ENUM_SERIALIZATION(target.h)

END()
