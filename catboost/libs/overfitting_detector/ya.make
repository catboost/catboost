LIBRARY()



PEERDIR(
    catboost/libs/logging
)

SRCS(
    overfitting_detector.cpp
)

GENERATE_ENUM_SERIALIZATION(overfitting_detector.h)

END()
