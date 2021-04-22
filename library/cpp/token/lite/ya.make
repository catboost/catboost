

LIBRARY()

SRCDIR(library/cpp/token)

SRCS(
    accent.cpp
    nlptypes.cpp
    token_util.cpp
)

GENERATE_ENUM_SERIALIZATION(nlptypes.h)

GENERATE_ENUM_SERIALIZATION(formtype.h)

PEERDIR(
    library/cpp/langmask
)

END()
