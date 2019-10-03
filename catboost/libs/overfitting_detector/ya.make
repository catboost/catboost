LIBRARY()



SRCS(
    error_tracker.cpp
    overfitting_detector.cpp
)

PEERDIR(
    catboost/libs/helpers
    catboost/libs/logging
    catboost/libs/metrics
    catboost/private/libs/options
    library/statistics
)

GENERATE_ENUM_SERIALIZATION(overfitting_detector.h)

END()
