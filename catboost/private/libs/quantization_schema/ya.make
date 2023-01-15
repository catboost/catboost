

LIBRARY()

SRCS(
    detail.cpp
    schema.cpp
    serialization.cpp
)

PEERDIR(
    catboost/idl/pool/proto
    catboost/libs/helpers
    catboost/private/libs/options
    catboost/private/libs/quantization
    library/cpp/json
)

GENERATE_ENUM_SERIALIZATION(serialization.h)

END()
