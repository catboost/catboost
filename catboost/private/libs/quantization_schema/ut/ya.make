

UNITTEST_FOR(catboost/private/libs/quantization_schema)

SRCS(
    serialization_ut.cpp
)

PEERDIR(
    catboost/idl/pool/proto
)

REQUIREMENTS(ram:12)

END()
