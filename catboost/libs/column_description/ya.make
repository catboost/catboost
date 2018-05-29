LIBRARY()



SRCS(
    cd_parser.cpp
)

PEERDIR(
    catboost/libs/data_util
    catboost/libs/helpers
    catboost/libs/logging
)

GENERATE_ENUM_SERIALIZATION(column.h)

END()
