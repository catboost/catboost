LIBRARY()



SRCS(
    cd_parser.cpp
    column.cpp
)

PEERDIR(
    library/binsaver
    catboost/libs/data_util
    catboost/libs/helpers
    catboost/libs/logging
)

GENERATE_ENUM_SERIALIZATION(column.h)

END()
