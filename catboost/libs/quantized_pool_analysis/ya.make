LIBRARY()

SRCS(
    quantized_pool_analysis.cpp
)

PEERDIR(
    catboost/libs/algo
    catboost/libs/data_new
    catboost/libs/model
    catboost/libs/target
)

END()
