LIBRARY()




SRCS(
    ctr_config.h
    ctr_type.cpp
)

PEERDIR(
    catboost/libs/helpers
)

GENERATE_ENUM_SERIALIZATION(ctr_type.h)

END()
