LIBRARY()

NO_UTIL()



SRCS(
    generated/composition.cpp
    generated/decomposition.cpp
    decomposition_table.h
    normalization.cpp
)

IF(NOT CATBOOST_OPENSOURCE)
    SRCS(
        custom_encoder.cpp
    )
    PEERDIR(
        library/charset
    )
    GENERATE_ENUM_SERIALIZATION(normalization.h)
ENDIF()

END()
