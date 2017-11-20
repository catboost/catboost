LIBRARY()



SRCS(
    cd_parser.cpp
)

PEERDIR(
    catboost/libs/helpers
    catboost/libs/logging
)

GENERATE_ENUM_SERIALIZATION(column.h)

END()
