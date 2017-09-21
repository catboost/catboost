LIBRARY()

NO_WERROR()



SRCS(
    full_model_saver.cpp
    model_converter.cpp
    final_mean_ctr.cpp
)

PEERDIR(
    catboost/cuda/data
    catboost/cuda/ctrs
    catboost/cuda/models
    catboost/libs/model
    catboost/libs/logging
    catboost/libs/data
    catboost/libs/algo
    library/grid_creator
)


END()
