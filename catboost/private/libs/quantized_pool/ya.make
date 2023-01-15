

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
    catboost/libs/data
    catboost/private/libs/data_util
    catboost/libs/helpers
    catboost/private/libs/index_range
    catboost/private/libs/labels
    catboost/libs/logging
    catboost/private/libs/quantization_schema
    catboost/private/libs/validate_fb
    contrib/libs/flatbuffers
    library/object_factory
)

GENERATE_ENUM_SERIALIZATION(print.h)

END()
