LIBRARY()



SRCS(
    eval_helpers.cpp
)

PEERDIR(
    catboost/libs/data
)

GENERATE_ENUM_SERIALIZATION(eval_helpers.h)

END()
