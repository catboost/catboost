LIBRARY()



SRCS(
    feature.cpp
    binarizations_manager.cpp
    permutation.cpp
    data_utils.cpp
)

PEERDIR(
    catboost/cuda/utils
    catboost/libs/ctr_description
    catboost/libs/data_new
    catboost/libs/data_util
    catboost/libs/labels
    catboost/libs/model
    catboost/libs/helpers
    catboost/libs/logging
    catboost/libs/options
    catboost/libs/pairs
    catboost/libs/quantization_schema
    catboost/libs/quantized_pool
    catboost/libs/quantization
    library/threading/local_executor
)

GENERATE_ENUM_SERIALIZATION(feature.h)

END()
