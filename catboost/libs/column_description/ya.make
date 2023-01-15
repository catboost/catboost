LIBRARY()



SRCS(
    cd_parser.cpp
    column.cpp
)

PEERDIR(
    library/cpp/binsaver
    catboost/private/libs/data_util
    catboost/libs/helpers
    catboost/libs/logging
)

GENERATE_ENUM_SERIALIZATION(column.h)

END()
