LIBRARY()

NO_WERROR()



SRCS(
    cpu_pool_based_data_provider_builder.cpp
    model_converter.cpp
)

PEERDIR(
    catboost/cuda/ctrs
    catboost/cuda/data
    catboost/cuda/models
    catboost/libs/data
    catboost/libs/helpers
    catboost/libs/logging
    catboost/libs/model
    catboost/libs/quantization
    library/json
)

END()
