

UNITTEST_FOR(catboost/private/libs/quantized_pool)

SIZE(MEDIUM)

SRCS(
    loader_ut.cpp
    serialization_ut.cpp
    print_ut.cpp
)

PEERDIR(
    catboost/idl/pool/flat
    catboost/libs/data
    catboost/libs/data/ut/lib
    catboost/private/libs/data_types
    catboost/private/libs/quantization_schema

    contrib/libs/flatbuffers

    library/json
)

END()
