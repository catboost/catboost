LIBRARY()



SRCS(
    data_provider.cpp
    load_data.cpp
    feature.cpp
    binarizations_manager.cpp
    permutation.cpp
    data_utils.cpp
    protobuf_data_provider_reader.cpp
)

PEERDIR(
    catboost/libs/ctr_description
    library/threading/local_executor
    library/grid_creator
    catboost/libs/helpers
    catboost/libs/logging
    catboost/libs/data
    catboost/libs/options
    catboost/cuda/data/pool_proto
    catboost/cuda/utils
)

GENERATE_ENUM_SERIALIZATION(columns.h)
GENERATE_ENUM_SERIALIZATION(feature.h)



END()
