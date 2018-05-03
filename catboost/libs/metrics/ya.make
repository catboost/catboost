LIBRARY()



SRCS(
    sample.cpp
    auc.cpp
    metric.cpp
    pfound.cpp
)

PEERDIR(
    catboost/libs/helpers
    library/containers/2d_array
    library/threading/local_executor
)

GENERATE_ENUM_SERIALIZATION(metric.h)

END()
