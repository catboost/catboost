LIBRARY()



SRCS(
    dataset.cpp
    GLOBAL doc_pool_data_provider.cpp
    load_data.cpp
    pool.cpp
    quantized_features.cpp
)

PEERDIR(
    catboost/libs/data_types
    catboost/libs/data_util
    catboost/libs/cat_feature
    catboost/libs/column_description
    catboost/libs/helpers
    catboost/libs/logging
    catboost/libs/pool_builder
    catboost/libs/quantization_schema
    catboost/libs/quantized_pool
    library/threading/future
    library/threading/local_executor
)

END()
