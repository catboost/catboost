

LIBRARY()

SRCS(
    detail.cpp
    GLOBAL loader.cpp
    pool.cpp
    print.cpp
    quantized.cpp
    serialization.cpp
)

PEERDIR(
    catboost/idl/pool/flat
    catboost/idl/pool/proto
    catboost/libs/column_description
    catboost/libs/data_new
    catboost/libs/data_util
    catboost/libs/helpers
    catboost/libs/logging
    catboost/libs/quantization_schema
    catboost/libs/validate_fb
    contrib/libs/flatbuffers
)

GENERATE_ENUM_SERIALIZATION(print.h)

END()
