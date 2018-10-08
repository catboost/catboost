LIBRARY()



SRCS(
    async_row_processor.cpp
    cat_feature_perfect_hash.cpp
    cat_feature_perfect_hash_helper.cpp
    GLOBAL cb_dsv_loader.cpp
    columns.cpp
    data_provider.cpp
    data_provider_builders.cpp
    external_columns.cpp
    feature_index.cpp
    features_layout.cpp
    load_data.cpp
    loader.cpp
    meta_info.cpp
    objects.cpp
    objects_grouping.cpp
    quantized_features_info.cpp
    target.cpp
    unaligned_mem.cpp
    util.cpp
    visitor.cpp
    weights.cpp
)

PEERDIR(
    library/dbg_output
    library/object_factory
    library/threading/future
    library/threading/local_executor

    catboost/libs/cat_feature
    catboost/libs/column_description
    catboost/libs/data_types
    catboost/libs/data_util
    catboost/libs/helpers
    catboost/libs/logging
    catboost/libs/model
    catboost/libs/options
    catboost/libs/quantization
    catboost/libs/quantization_schema
)

GENERATE_ENUM_SERIALIZATION(objects.h)
GENERATE_ENUM_SERIALIZATION(target.h)
GENERATE_ENUM_SERIALIZATION(visitor.h)

END()
