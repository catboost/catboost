

LIBRARY()

SRCS(
    data_provider.cpp
    dsv_parser.cpp
)

PEERDIR(
    catboost/libs/column_description
    catboost/libs/data_new
    catboost/libs/helpers
)

GENERATE_ENUM_SERIALIZATION(dsv_parser.h)

END()
