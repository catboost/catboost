

UNITTEST_FOR(catboost/private/libs/quantization_schema)

SRCS(
    quantize_ut.cpp
    serialization_ut.cpp
)

PEERDIR(
    catboost/idl/pool/proto
)

END()
