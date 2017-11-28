LIBRARY()



SRCS(
    overfitting_detector.cpp
)

PEERDIR(
    catboost/libs/helpers
    catboost/libs/logging
    catboost/libs/options
    library/logger/global
    library/statistics
)

GENERATE_ENUM_SERIALIZATION(overfitting_detector.h)

END()
