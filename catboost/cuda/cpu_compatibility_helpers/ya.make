LIBRARY()



SRCS(
    model_converter.cpp
)

PEERDIR(
    catboost/cuda/data
    catboost/cuda/models
    catboost/libs/algo
    catboost/libs/data
    catboost/libs/helpers
    catboost/libs/model
    catboost/libs/target
)

END()
