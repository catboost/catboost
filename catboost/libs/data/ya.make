LIBRARY()



SRCS(
    async_row_processor.h
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
    catboost/libs/model
    catboost/libs/pool_builder
    library/threading/future
    library/threading/local_executor
)

END()
