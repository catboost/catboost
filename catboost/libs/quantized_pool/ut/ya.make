

UNITTEST_FOR(catboost/libs/quantized_pool)

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
    catboost/libs/data_types
    catboost/libs/quantization_schema

    contrib/libs/flatbuffers
)

END()
