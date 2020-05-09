

LIBRARY()

SRCS(
    charfilter.cpp
    nlptypes.cpp
    token_util.cpp
    generated/custom_decompositions.cpp
)

DEFAULT(USE_NEW_DECOMPOSITION_TABLE no)

IF (USE_NEW_DECOMPOSITION_TABLE)
    SET_APPEND(CFLAGS -DUSE_NEW_DECOMPOSITION_TABLE)
ENDIF()

GENERATE_ENUM_SERIALIZATION(nlptypes.h)

GENERATE_ENUM_SERIALIZATION(formtype.h)

PEERDIR(
    library/langmask
    library/cpp/unicode/normalization
)

END()
