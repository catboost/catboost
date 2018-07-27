

LIBRARY()

SRCS(
    detail.cpp
    pool.cpp
    print.cpp
    serialization.cpp
)

PEERDIR(
    catboost/idl/pool/flat
    catboost/idl/pool/proto
    catboost/libs/column_description
    catboost/libs/validate_fb
    contrib/libs/flatbuffers
)

GENERATE_ENUM_SERIALIZATION(print.h)

END()
