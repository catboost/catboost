

LIBRARY()

SRCS(
    decomposition.cpp
    generated/custom_decompositions.cpp
)

GENERATE_ENUM_SERIALIZATION(nlptypes.h)

GENERATE_ENUM_SERIALIZATION(formtype.h)

PEERDIR(
    library/cpp/token/lite
    library/cpp/unicode/normalization
)

END()
