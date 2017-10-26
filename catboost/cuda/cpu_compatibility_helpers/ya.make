LIBRARY()

NO_WERROR()



SRCS(
    cpu_pool_based_data_provider_builder.cpp
    full_model_saver.cpp
    model_converter.cpp
    final_mean_ctr.cpp
)

PEERDIR(
    catboost/cuda/data
    catboost/cuda/ctrs
    catboost/cuda/models
    catboost/libs/logging
    catboost/libs/data
    catboost/libs/algo
    library/json
    library/grid_creator
)

END()
