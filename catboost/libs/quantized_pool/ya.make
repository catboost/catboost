

LIBRARY()

SRCS(
    pool.cpp
    print.cpp
    serialization.cpp
)

PEERDIR(
    catboost/idl/pool/flat
    catboost/libs/validate_fb
    contrib/libs/flatbuffers
)

END()
