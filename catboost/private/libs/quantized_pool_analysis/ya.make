LIBRARY()



SRCS(
    quantized_pool_analysis.cpp
)

PEERDIR(
    catboost/private/libs/algo
    catboost/libs/data
    catboost/libs/model
    catboost/private/libs/target
)

END()
