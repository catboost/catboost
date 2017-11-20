LIBRARY()



SRCS(
    params.cpp
)

PEERDIR(
    catboost/libs/helpers
    catboost/libs/metrics
    catboost/libs/model
    catboost/libs/overfitting_detector
    library/binsaver
    library/grid_creator
    library/json
)

GENERATE_ENUM_SERIALIZATION(params.h)

END()
