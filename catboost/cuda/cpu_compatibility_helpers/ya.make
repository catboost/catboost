LIBRARY()

NO_WERROR()



SRCS(
    cpu_pool_based_data_provider_builder.cpp
    full_model_saver.cpp
    model_converter.cpp
)

PEERDIR(
    catboost/cuda/ctrs
    catboost/cuda/data
    catboost/cuda/models
    catboost/libs/data
    catboost/libs/logging
    catboost/libs/model
    library/grid_creator
    library/json
)

END()
